# Z-Probe support
#
# Copyright (C) 2017-2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import pins
from . import manual_probe

HINT_TIMEOUT = """
If the probe did not move far enough to trigger, then
consider reducing the Z axis minimum position so the probe
can travel further (the Z minimum position can be negative).
"""

# Calculate the average Z from a set of positions
def calc_probe_z_average(positions, method='average'):
    if method != 'median':
        # Use mean average
        count = float(len(positions))
        return [sum([pos[i] for pos in positions]) / count
                for i in range(3)]
    # Use median
    z_sorted = sorted(positions, key=(lambda p: p[2]))
    middle = len(positions) // 2
    if (len(positions) & 1) == 1:
        # odd number of samples
        return z_sorted[middle]
    # even number of samples
    return calc_probe_z_average(z_sorted[middle-1:middle+1], 'average')


######################################################################
# Probe device implementation helpers
######################################################################

# Helper to implement common probing commands
class ProbeCommandHelper:
    def __init__(self, config, probe, query_endstop=None):
        self.printer = config.get_printer()
        self.probe = probe
        self.query_endstop = query_endstop
        self.name = config.get_name()
        gcode = self.printer.lookup_object('gcode')
        # QUERY_PROBE command
        self.last_state = False
        gcode.register_command('QUERY_PROBE', self.cmd_QUERY_PROBE,
                               desc=self.cmd_QUERY_PROBE_help)
        # PROBE command
        self.last_z_result = 0.
        gcode.register_command('PROBE', self.cmd_PROBE,
                               desc=self.cmd_PROBE_help)
        # PROBE_CALIBRATE command
        self.probe_calibrate_z = 0.
        gcode.register_command('PROBE_CALIBRATE', self.cmd_PROBE_CALIBRATE,
                               desc=self.cmd_PROBE_CALIBRATE_help)
        # Other commands
        gcode.register_command('PROBE_ACCURACY', self.cmd_PROBE_ACCURACY,
                               desc=self.cmd_PROBE_ACCURACY_help)
        gcode.register_command('Z_OFFSET_APPLY_PROBE',
                               self.cmd_Z_OFFSET_APPLY_PROBE,
                               desc=self.cmd_Z_OFFSET_APPLY_PROBE_help)
        # SET_ACTIVE_PROBE command
        gcode.register_command('SET_ACTIVE_PROBE', self.cmd_SET_ACTIVE_PROBE,
                               desc=self.cmd_SET_ACTIVE_PROBE_help)
        gcode.register_command('CALIBRATE_DUAL_Z_OFFSET', self.cmd_CALIBRATE_DUAL_Z_OFFSET,
                               desc=self.cmd_CALIBRATE_DUAL_Z_OFFSET_help)
    def _move(self, coord, speed):
        self.printer.lookup_object('toolhead').manual_move(coord, speed)
    def get_status(self, eventtime):
        # The 'name' here refers to the config section name this helper was initialized with.
        # If PrinterProbe was initialized with main config, this 'name' might be [probe]
        # or the first [probe_tool*] section. This might be fine, or might need adjustment
        # if status reports need to reflect the *currently active* probe's specific name.
        # For now, keeping it as is.
        return {'name': self.name, 
                'active_probe_name': self.probe.active_probe_name if hasattr(self.probe, 'active_probe_name') else None,
                'last_query': self.last_state,
                'last_z_result': self.last_z_result}
    cmd_SET_ACTIVE_PROBE_help = "Set the active probe by its configured name"
    def cmd_SET_ACTIVE_PROBE(self, gcmd):
        probe_name = gcmd.get_command_parameter("PROBE")
        if not probe_name:
            raise gcmd.error("SET_ACTIVE_PROBE requires PROBE=<name> parameter.")
        
        # Access the _set_active_probe method from the main PrinterProbe instance
        if not hasattr(self.probe, '_set_active_probe'):
            raise gcmd.error("Active probe switching is not supported by the current probe setup.")
            
        self.probe._set_active_probe(probe_name)
        gcmd.respond_info("Active probe set to '%s'" % probe_name)

    cmd_CALIBRATE_DUAL_Z_OFFSET_help = "Calibrates the Z offset between two probes on a dual carriage/IDEX printer."
    def cmd_CALIBRATE_DUAL_Z_OFFSET(self,gcmd):
        x_pos = gcmd.get_float("X")
        y_pos = gcmd.get_float("Y")
        if x_pos is None or y_pos is None:
            raise gcmd.error("CALIBRATE_DUAL_Z_OFFSET requires X and Y parameters.")

        primary_probe_name = gcmd.get("PRIMARY_PROBE_NAME", "0")
        secondary_probe_name = gcmd.get("SECONDARY_PROBE_NAME", "1")
        primary_extruder_name = gcmd.get("PRIMARY_EXTRUDER_NAME", "extruder")
        secondary_extruder_name = gcmd.get("SECONDARY_EXTRUDER_NAME", "extruder1")

        if primary_probe_name == secondary_probe_name:
            raise gcmd.error("PRIMARY_PROBE_NAME and SECONDARY_PROBE_NAME must be different.")

        if primary_probe_name not in self.probe.probe_endstops:
            raise gcmd.error(f"Primary probe '{primary_probe_name}' not found.")
        if secondary_probe_name not in self.probe.probe_endstops:
            raise gcmd.error(f"Secondary probe '{secondary_probe_name}' not found.")

        gcode = self.printer.lookup_object('gcode')
        toolhead = self.printer.lookup_object('toolhead')
        
        lift_height = 5.0 # Standard lift height
        
        # Create a dummy gcmd for run_single_probe to avoid param conflicts
        # and use default probe parameters.
        dummy_gcmd = gcode.create_gcode_command("", "", {})

        gcode.run_script_from_command("SAVE_GCODE_STATE NAME=calibrate_dual_z_offset_internal")
        try:
            gcmd.respond_info("Starting dual Z offset calibration...")
            gcmd.respond_info(f"Primary Probe: {primary_probe_name}, Secondary Probe: {secondary_probe_name}")
            gcmd.respond_info(f"Primary Extruder: {primary_extruder_name}, Secondary Extruder: {secondary_extruder_name}")
            gcmd.respond_info(f"Probing at X={x_pos}, Y={y_pos}")

            gcode.run_script_from_command("G28")

            # --- Primary Probe Sequence ---
            gcmd.respond_info(f"Activating primary extruder: {primary_extruder_name}")
            gcode.run_script_from_command(f"ACTIVATE_EXTRUDER EXTRUDER={primary_extruder_name}")
            
            # Verify probe switch (event handler should have done this)
            # A short delay might be needed if event processing isn't immediate enough,
            # but Klipper's gcode interpreter typically waits for event handlers.
            if self.probe.active_probe_name != primary_probe_name:
                # Attempt to set it manually if auto switch failed or was unexpected
                gcmd.respond_info(f"Active probe is {self.probe.active_probe_name}, expected {primary_probe_name}. Attempting manual switch...")
                self.probe._set_active_probe(primary_probe_name) # Try to force it
                if self.probe.active_probe_name != primary_probe_name:
                    raise gcmd.error(
                        f"Failed to switch to primary probe '{primary_probe_name}'. "
                        f"Current active probe: '{self.probe.active_probe_name}'.")
            
            gcmd.respond_info(f"Using primary probe: {self.probe.active_probe_name}")
            
            # Lift before moving to XY
            current_pos_primary = toolhead.get_position()
            primary_probe_params = self.probe.get_probe_params(dummy_gcmd) # Get lift_speed
            self._move([None, None, current_pos_primary[2] + lift_height], primary_probe_params['lift_speed'])
            
            # Move to XY
            self._move([x_pos, y_pos, None], primary_probe_params['probe_speed']) # Use probe_speed for travel to point
            
            gcmd.respond_info("Probing with primary probe...")
            pos_primary = run_single_probe(self.probe, dummy_gcmd)
            gcmd.respond_info(f"Primary probe result: Z={pos_primary[2]:.6f}")

            # Lift Z after primary probe
            self._move([None, None, pos_primary[2] + lift_height], primary_probe_params['lift_speed'])

            # --- Secondary Probe Sequence ---
            gcmd.respond_info(f"Activating secondary extruder: {secondary_extruder_name}")
            gcode.run_script_from_command(f"ACTIVATE_EXTRUDER EXTRUDER={secondary_extruder_name}")

            if self.probe.active_probe_name != secondary_probe_name:
                gcmd.respond_info(f"Active probe is {self.probe.active_probe_name}, expected {secondary_probe_name}. Attempting manual switch...")
                self.probe._set_active_probe(secondary_probe_name)
                if self.probe.active_probe_name != secondary_probe_name:
                    raise gcmd.error(
                        f"Failed to switch to secondary probe '{secondary_probe_name}'. "
                        f"Current active probe: '{self.probe.active_probe_name}'.")

            gcmd.respond_info(f"Using secondary probe: {self.probe.active_probe_name}")
            
            secondary_probe_params = self.probe.get_probe_params(dummy_gcmd)
            current_pos_secondary = toolhead.get_position() # Get current position which might be different
            # Lift Z again before moving to XY, relative to current Z
            self._move([None, None, current_pos_secondary[2] + lift_height], secondary_probe_params['lift_speed'])
            
            # Move to XY
            self._move([x_pos, y_pos, None], secondary_probe_params['probe_speed'])
            
            gcmd.respond_info("Probing with secondary probe...")
            pos_secondary = run_single_probe(self.probe, dummy_gcmd)
            gcmd.respond_info(f"Secondary probe result: Z={pos_secondary[2]:.6f}")

            # --- Calculate and Apply Offset ---
            trigger_z_primary = pos_primary[2]
            trigger_z_secondary = pos_secondary[2]
            measured_trigger_difference = trigger_z_secondary - trigger_z_primary
            
            gcmd.respond_info(f"Measured Z trigger difference (secondary - primary): {measured_trigger_difference:.6f} mm")

            current_secondary_z_offset = self.probe.probe_offsets_helpers[secondary_probe_name].z_offset
            # If T1's probe triggers lower (more negative measured_trigger_difference),
            # this means T1's nozzle is effectively higher than T0's for the same probe trigger point,
            # or its probe is "longer". To compensate, we want to make T1's nozzle go lower,
            # which means increasing its z_offset (making it less negative or more positive).
            # So, new_offset = current_offset - difference
            new_secondary_z_offset = current_secondary_z_offset - measured_trigger_difference
            
            secondary_probe_config_section_name = self.probe.probe_endstops[
                secondary_probe_name].get_config_section_name()

            configfile = self.printer.lookup_object('configfile')
            configfile.set(secondary_probe_config_section_name, 'z_offset', f"{new_secondary_z_offset:.3f}")

            gcmd.respond_info(
                f"Adjusted z_offset for probe '{secondary_probe_name}' (section '{secondary_probe_config_section_name}')\n"
                f"from {current_secondary_z_offset:.3f} to {new_secondary_z_offset:.3f}\n"
                "Run SAVE_CONFIG to make this change permanent.")

        finally:
            gcode.run_script_from_command("RESTORE_GCODE_STATE NAME=calibrate_dual_z_offset_internal")
            gcmd.respond_info("Dual Z offset calibration finished or aborted.")
        
        pass # End of command

    cmd_QUERY_PROBE_help = "Return the status of the z-probe"
    def cmd_QUERY_PROBE(self, gcmd):
        if self.query_endstop is None:
            raise gcmd.error("Probe does not support QUERY_PROBE")
        toolhead = self.printer.lookup_object('toolhead')
        print_time = toolhead.get_last_move_time()
        res = self.query_endstop(print_time)
        self.last_state = res
        gcmd.respond_info("probe: %s" % (["open", "TRIGGERED"][not not res],))
    cmd_PROBE_help = "Probe Z-height at current XY position"
    def cmd_PROBE(self, gcmd):
        pos = run_single_probe(self.probe, gcmd)
        gcmd.respond_info("Result is z=%.6f" % (pos[2],))
        self.last_z_result = pos[2]
    def probe_calibrate_finalize(self, kin_pos):
        if kin_pos is None:
            return
        z_offset = self.probe_calibrate_z - kin_pos[2]
        gcode = self.printer.lookup_object('gcode')
        active_probe_section_name = self.probe.probe_endstops[
            self.probe.active_probe_name].get_config_section_name()
        gcode.respond_info(
            "Probe %s: z_offset: %.3f\n"
            "The SAVE_CONFIG command will update the printer config file\n"
            "with the above and restart the printer." % (active_probe_section_name, z_offset))
        configfile = self.printer.lookup_object('configfile')
        configfile.set(active_probe_section_name, 'z_offset', "%.3f" % (z_offset,))

    # Helper for automatic probe calibration when using an eddy current probe
    def _auto_eddy_probe(self, start_pos, speed, threshold=0.01,
                         settle_samples=3, step=0.02):
        toolhead = self.printer.lookup_object('toolhead')
        sensor = getattr(self.probe, 'sensor_helper', None)
        if sensor is None:
            raise self.printer.command_error(
                "Automatic eddy probe calibration requires an eddy current probe")
        last_val = [None]
        stable = [0]
        stop_flag = [False]

        def handle_batch(msg):
            if stop_flag[0]:
                return False
            for _, freq, _ in msg['data']:
                if last_val[0] is not None and abs(freq - last_val[0]) < threshold:
                    stable[0] += 1
                else:
                    stable[0] = 0
                last_val[0] = freq
                if stable[0] >= settle_samples:
                    stop_flag[0] = True
                    return False
            return True

        sensor.add_client(handle_batch)
        curpos = list(start_pos)
        while not stop_flag[0]:
            curpos[2] -= step
            self._move(curpos, speed)
            toolhead.dwell(0.050)
        toolhead.wait_moves()
        kin = toolhead.get_kinematics()
        kin_spos = {s.get_name(): s.get_commanded_position() for s in kin.get_steppers()}
        return kin.calc_position(kin_spos)
    cmd_PROBE_CALIBRATE_help = "Calibrate the probe's z_offset"
    def cmd_PROBE_CALIBRATE(self, gcmd):
        manual_probe.verify_no_manual_probe(self.printer)
        
        # Get the tap_direct_z_offset flag for the active probe
        active_probe_params = self.probe.active_param_helper 
        is_tap_direct_mode = active_probe_params.tap_direct_z_offset

        if is_tap_direct_mode:
            gcmd.respond_info("TAP direct Z offset mode is active for probe '%s'."
                              % self.probe.active_probe_name)
            
            # Perform initial probe
            curpos = run_single_probe(self.probe, gcmd)
            self.probe_calibrate_z = curpos[2] # Store the trigger Z position

            # Set z_offset to 0.000
            active_probe_section_name = self.probe.probe_endstops[
                self.probe.active_probe_name].get_config_section_name()
            
            # gcode object is already available via self.printer.lookup_object('gcode')
            # but gcmd.respond_info is sufficient for messages.
            gcmd.respond_info(
                "Probe %s: z_offset set to 0.000 due to TAP direct mode.\n"
                "The SAVE_CONFIG command will update the printer config file\n"
                "with the above and restart the printer." % (active_probe_section_name,))
            
            configfile = self.printer.lookup_object('configfile')
            configfile.set(active_probe_section_name, 'z_offset', "0.000")
            
            # Skip manual probe steps. The original code for moving nozzle and 
            # calling ManualProbeHelper is omitted in this branch.
        else:
            params = self.probe.get_probe_params(gcmd)
            curpos = run_single_probe(self.probe, gcmd)
            self.probe_calibrate_z = curpos[2]
            curpos[2] += 5.
            self._move(curpos, params['lift_speed'])
            x_offset, y_offset, z_unused = self.probe.get_offsets()
            curpos[0] += x_offset
            curpos[1] += y_offset
            self._move(curpos, params['probe_speed'])
            if hasattr(self.probe, 'sensor_helper'):
                kin_pos = self._auto_eddy_probe(curpos, params['probe_speed'])
                self.probe_calibrate_finalize(kin_pos)
            else:
                manual_probe.ManualProbeHelper(self.printer, gcmd,
                                               self.probe_calibrate_finalize)
    cmd_PROBE_ACCURACY_help = "Probe Z-height accuracy at current XY position"
    def cmd_PROBE_ACCURACY(self, gcmd):
        params = self.probe.get_probe_params(gcmd)
        sample_count = gcmd.get_int("SAMPLES", 10, minval=1)
        toolhead = self.printer.lookup_object('toolhead')
        pos = toolhead.get_position()
        gcmd.respond_info("PROBE_ACCURACY at X:%.3f Y:%.3f Z:%.3f"
                          " (samples=%d retract=%.3f"
                          " speed=%.1f lift_speed=%.1f)\n"
                          % (pos[0], pos[1], pos[2],
                             sample_count, params['sample_retract_dist'],
                             params['probe_speed'], params['lift_speed']))
        # Create dummy gcmd with SAMPLES=1
        fo_params = dict(gcmd.get_command_parameters())
        fo_params['SAMPLES'] = '1'
        gcode = self.printer.lookup_object('gcode')
        fo_gcmd = gcode.create_gcode_command("", "", fo_params)
        # Probe bed sample_count times
        probe_session = self.probe.start_probe_session(fo_gcmd)
        probe_num = 0
        while probe_num < sample_count:
            # Probe position
            probe_session.run_probe(fo_gcmd)
            probe_num += 1
            # Retract
            pos = toolhead.get_position()
            liftpos = [None, None, pos[2] + params['sample_retract_dist']]
            self._move(liftpos, params['lift_speed'])
        positions = probe_session.pull_probed_results()
        probe_session.end_probe_session()
        # Calculate maximum, minimum and average values
        max_value = max([p[2] for p in positions])
        min_value = min([p[2] for p in positions])
        range_value = max_value - min_value
        avg_value = calc_probe_z_average(positions, 'average')[2]
        median = calc_probe_z_average(positions, 'median')[2]
        # calculate the standard deviation
        deviation_sum = 0
        for i in range(len(positions)):
            deviation_sum += pow(positions[i][2] - avg_value, 2.)
        sigma = (deviation_sum / len(positions)) ** 0.5
        # Show information
        gcmd.respond_info(
            "probe accuracy results: maximum %.6f, minimum %.6f, range %.6f, "
            "average %.6f, median %.6f, standard deviation %.6f" % (
            max_value, min_value, range_value, avg_value, median, sigma))
    cmd_Z_OFFSET_APPLY_PROBE_help = "Adjust the probe's z_offset"
    def cmd_Z_OFFSET_APPLY_PROBE(self, gcmd):
        gcode_move = self.printer.lookup_object("gcode_move")
        offset = gcode_move.get_status()['homing_origin'].z
        if offset == 0:
            gcmd.respond_info("Nothing to do: Z Offset is 0")
            return
        z_offset = self.probe.get_offsets()[2]
        new_calibrate = z_offset - offset
        active_probe_section_name = self.probe.probe_endstops[
            self.probe.active_probe_name].get_config_section_name()
        gcmd.respond_info(
            "Probe %s: z_offset: %.3f\n"
            "The SAVE_CONFIG command will update the printer config file\n"
            "with the above and restart the printer."
            % (active_probe_section_name, new_calibrate))
        configfile = self.printer.lookup_object('configfile')
        configfile.set(active_probe_section_name, 'z_offset', "%.3f" % (new_calibrate,))

# Helper to lookup the minimum Z position for the printer
def lookup_minimum_z(config):
    zconfig = manual_probe.lookup_z_endstop_config(config)
    if zconfig is not None:
        return zconfig.getfloat('position_min', 0., note_valid=False)
    pconfig = config.getsection('printer')
    return pconfig.getfloat('minimum_z_position', 0., note_valid=False)

# Helper to lookup all the Z axis steppers
class LookupZSteppers:
    def __init__(self, config, add_stepper_cb):
        self.printer = config.get_printer()
        self.add_stepper_cb = add_stepper_cb
        self.printer.register_event_handler('klippy:mcu_identify',
                                            self._handle_mcu_identify)
    def _handle_mcu_identify(self):
        kin = self.printer.lookup_object('toolhead').get_kinematics()
        for stepper in kin.get_steppers():
            if stepper.is_active_axis('z'):
                self.add_stepper_cb(stepper)

# Homing via probe:z_virtual_endstop
class HomingViaProbeHelper:
    def __init__(self, config, mcu_probe, param_helper):
        self.printer = config.get_printer()
        self.mcu_probe = mcu_probe
        self.param_helper = param_helper
        self.multi_probe_pending = False
        self.z_min_position = lookup_minimum_z(config)
        self.results = []
        LookupZSteppers(config, self.mcu_probe.add_stepper)
        # Register z_virtual_endstop pin
        self.printer.lookup_object('pins').register_chip('probe', self)
        # Register event handlers
        self.printer.register_event_handler("homing:homing_move_begin",
                                            self._handle_homing_move_begin)
        self.printer.register_event_handler("homing:homing_move_end",
                                            self._handle_homing_move_end)
        self.printer.register_event_handler("homing:home_rails_begin",
                                            self._handle_home_rails_begin)
        self.printer.register_event_handler("homing:home_rails_end",
                                            self._handle_home_rails_end)
        self.printer.register_event_handler("gcode:command_error",
                                            self._handle_command_error)
    def _handle_homing_move_begin(self, hmove):
        if self.mcu_probe in hmove.get_mcu_endstops():
            self.mcu_probe.probe_prepare(hmove)
    def _handle_homing_move_end(self, hmove):
        if self.mcu_probe in hmove.get_mcu_endstops():
            self.mcu_probe.probe_finish(hmove)
    def _handle_home_rails_begin(self, homing_state, rails):
        endstops = [es for rail in rails for es, name in rail.get_endstops()]
        if self.mcu_probe in endstops:
            self.mcu_probe.multi_probe_begin()
            self.multi_probe_pending = True
    def _handle_home_rails_end(self, homing_state, rails):
        endstops = [es for rail in rails for es, name in rail.get_endstops()]
        if self.multi_probe_pending and self.mcu_probe in endstops:
            self.multi_probe_pending = False
            self.mcu_probe.multi_probe_end()
    def _handle_command_error(self):
        if self.multi_probe_pending:
            self.multi_probe_pending = False
            try:
                self.mcu_probe.multi_probe_end()
            except:
                logging.exception("Homing multi-probe end")
    def setup_pin(self, pin_type, pin_params):
        if pin_type != 'endstop' or pin_params['pin'] != 'z_virtual_endstop':
            raise pins.error("Probe virtual endstop only useful as endstop pin")
        if pin_params['invert'] or pin_params['pullup']:
            raise pins.error("Can not pullup/invert probe virtual endstop")
        return self.mcu_probe
    # Helper to convert probe based commands to use homing module
    def start_probe_session(self, gcmd):
        self.mcu_probe.multi_probe_begin()
        self.results = []
        return self
    def run_probe(self, gcmd):
        toolhead = self.printer.lookup_object('toolhead')
        pos = toolhead.get_position()
        pos[2] = self.z_min_position
        speed = self.param_helper.get_probe_params(gcmd)['probe_speed']
        phoming = self.printer.lookup_object('homing')
        self.results.append(phoming.probing_move(self.mcu_probe, pos, speed))
    def pull_probed_results(self):
        res = self.results
        self.results = []
        return res
    def end_probe_session(self):
        self.results = []
        self.mcu_probe.multi_probe_end()

# Helper to read multi-sample parameters from config
class ProbeParameterHelper:
    def __init__(self, config):
        gcode = config.get_printer().lookup_object('gcode')
        self.dummy_gcode_cmd = gcode.create_gcode_command("", "", {})
        # Configurable probing speeds
        self.speed = config.getfloat('speed', 5.0, above=0.)
        self.lift_speed = config.getfloat('lift_speed', self.speed, above=0.)
        # Multi-sample support (for improved accuracy)
        self.sample_count = config.getint('samples', 1, minval=1)
        self.sample_retract_dist = config.getfloat('sample_retract_dist', 2.,
                                                   above=0.)
        atypes = ['median', 'average']
        self.samples_result = config.getchoice('samples_result', atypes,
                                               'average')
        self.samples_tolerance = config.getfloat('samples_tolerance', 0.100,
                                                 minval=0.)
        self.samples_retries = config.getint('samples_tolerance_retries', 0,
                                             minval=0)
        # New parameter
        self.tap_direct_z_offset = config.getboolean('tap_direct_z_offset', False)
    def get_probe_params(self, gcmd=None):
        if gcmd is None:
            gcmd = self.dummy_gcode_cmd
        probe_speed = gcmd.get_float("PROBE_SPEED", self.speed, above=0.)
        lift_speed = gcmd.get_float("LIFT_SPEED", self.lift_speed, above=0.)
        samples = gcmd.get_int("SAMPLES", self.sample_count, minval=1)
        sample_retract_dist = gcmd.get_float("SAMPLE_RETRACT_DIST",
                                             self.sample_retract_dist, above=0.)
        samples_tolerance = gcmd.get_float("SAMPLES_TOLERANCE",
                                           self.samples_tolerance, minval=0.)
        samples_retries = gcmd.get_int("SAMPLES_TOLERANCE_RETRIES",
                                       self.samples_retries, minval=0)
        samples_result = gcmd.get("SAMPLES_RESULT", self.samples_result)
        return {'probe_speed': probe_speed,
                'lift_speed': lift_speed,
                'samples': samples,
                'sample_retract_dist': sample_retract_dist,
                'samples_tolerance': samples_tolerance,
                'samples_tolerance_retries': samples_retries,
                'samples_result': samples_result}

# Helper to track multiple probe attempts in a single command
class ProbeSessionHelper:
    def __init__(self, config, param_helper, start_session_cb):
        self.printer = config.get_printer()
        self.param_helper = param_helper
        self.start_session_cb = start_session_cb
        # Session state
        self.hw_probe_session = None
        self.results = []
        # Register event handlers
        self.printer.register_event_handler("gcode:command_error",
                                            self._handle_command_error)
    def _handle_command_error(self):
        if self.hw_probe_session is not None:
            try:
                self.end_probe_session()
            except:
                logging.exception("Multi-probe end")
    def _probe_state_error(self):
        raise self.printer.command_error(
            "Internal probe error - start/end probe session mismatch")
    def start_probe_session(self, gcmd):
        if self.hw_probe_session is not None:
            self._probe_state_error()
        self.hw_probe_session = self.start_session_cb(gcmd)
        self.results = []
        return self
    def end_probe_session(self):
        hw_probe_session = self.hw_probe_session
        if hw_probe_session is None:
            self._probe_state_error()
        self.results = []
        self.hw_probe_session = None
        hw_probe_session.end_probe_session()
    def _probe(self, gcmd):
        toolhead = self.printer.lookup_object('toolhead')
        curtime = self.printer.get_reactor().monotonic()
        if 'z' not in toolhead.get_status(curtime)['homed_axes']:
            raise self.printer.command_error("Must home before probe")
        try:
            self.hw_probe_session.run_probe(gcmd)
            epos = self.hw_probe_session.pull_probed_results()[0]
        except self.printer.command_error as e:
            reason = str(e)
            if "Timeout during endstop homing" in reason:
                reason += HINT_TIMEOUT
            raise self.printer.command_error(reason)
        # Allow axis_twist_compensation to update results
        self.printer.send_event("probe:update_results", epos)
        # Report results
        gcode = self.printer.lookup_object('gcode')
        gcode.respond_info("probe at %.3f,%.3f is z=%.6f"
                           % (epos[0], epos[1], epos[2]))
        return epos[:3]
    def run_probe(self, gcmd):
        if self.hw_probe_session is None:
            self._probe_state_error()
        params = self.param_helper.get_probe_params(gcmd)
        toolhead = self.printer.lookup_object('toolhead')
        probexy = toolhead.get_position()[:2]
        retries = 0
        positions = []
        sample_count = params['samples']
        while len(positions) < sample_count:
            # Probe position
            pos = self._probe(gcmd)
            positions.append(pos)
            # Check samples tolerance
            z_positions = [p[2] for p in positions]
            if max(z_positions)-min(z_positions) > params['samples_tolerance']:
                if retries >= params['samples_tolerance_retries']:
                    raise gcmd.error("Probe samples exceed samples_tolerance")
                gcmd.respond_info("Probe samples exceed tolerance. Retrying...")
                retries += 1
                positions = []
            # Retract
            if len(positions) < sample_count:
                toolhead.manual_move(
                    probexy + [pos[2] + params['sample_retract_dist']],
                    params['lift_speed'])
        # Calculate result
        epos = calc_probe_z_average(positions, params['samples_result'])
        self.results.append(epos)
    def pull_probed_results(self):
        res = self.results
        self.results = []
        return res

# Helper to read the xyz probe offsets from the config
class ProbeOffsetsHelper:
    def __init__(self, config):
        self.x_offset = config.getfloat('x_offset', 0.)
        self.y_offset = config.getfloat('y_offset', 0.)
        self.z_offset = config.getfloat('z_offset')
    def get_offsets(self):
        return self.x_offset, self.y_offset, self.z_offset


######################################################################
# Tools for utilizing the probe
######################################################################

# Helper code that can probe a series of points and report the
# position at each point.
class ProbePointsHelper:
    def __init__(self, config, finalize_callback, default_points=None):
        self.printer = config.get_printer()
        self.finalize_callback = finalize_callback
        self.probe_points = default_points
        self.name = config.get_name()
        self.gcode = self.printer.lookup_object('gcode')
        # Read config settings
        if default_points is None or config.get('points', None) is not None:
            self.probe_points = config.getlists('points', seps=(',', '\n'),
                                                parser=float, count=2)
        def_move_z = config.getfloat('horizontal_move_z', 5.)
        self.default_horizontal_move_z = def_move_z
        self.speed = config.getfloat('speed', 50., above=0.)
        self.use_offsets = False
        # Internal probing state
        self.lift_speed = self.speed
        self.probe_offsets = (0., 0., 0.)
        self.manual_results = []
    def minimum_points(self,n):
        if len(self.probe_points) < n:
            raise self.printer.config_error(
                "Need at least %d probe points for %s" % (n, self.name))
    def update_probe_points(self, points, min_points):
        self.probe_points = points
        self.minimum_points(min_points)
    def use_xy_offsets(self, use_offsets):
        self.use_offsets = use_offsets
    def get_lift_speed(self):
        return self.lift_speed
    def _move(self, coord, speed):
        self.printer.lookup_object('toolhead').manual_move(coord, speed)
    def _raise_tool(self, is_first=False):
        speed = self.lift_speed
        if is_first:
            # Use full speed to first probe position
            speed = self.speed
        self._move([None, None, self.horizontal_move_z], speed)
    def _invoke_callback(self, results):
        # Flush lookahead queue
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.get_last_move_time()
        # Invoke callback
        res = self.finalize_callback(self.probe_offsets, results)
        return res != "retry"
    def _move_next(self, probe_num):
        # Move to next XY probe point
        nextpos = list(self.probe_points[probe_num])
        if self.use_offsets:
            nextpos[0] -= self.probe_offsets[0]
            nextpos[1] -= self.probe_offsets[1]
        self._move(nextpos, self.speed)
    def start_probe(self, gcmd):
        manual_probe.verify_no_manual_probe(self.printer)
        # Lookup objects
        probe = self.printer.lookup_object('probe', None)
        method = gcmd.get('METHOD', 'automatic').lower()
        def_move_z = self.default_horizontal_move_z
        self.horizontal_move_z = gcmd.get_float('HORIZONTAL_MOVE_Z',
                                                def_move_z)
        if probe is None or method == 'manual':
            # Manual probe
            self.lift_speed = self.speed
            self.probe_offsets = (0., 0., 0.)
            self.manual_results = []
            self._manual_probe_start()
            return
        # Perform automatic probing
        self.lift_speed = probe.get_probe_params(gcmd)['lift_speed']
        self.probe_offsets = probe.get_offsets()
        if self.horizontal_move_z < self.probe_offsets[2]:
            raise gcmd.error("horizontal_move_z can't be less than"
                             " probe's z_offset")
        probe_session = probe.start_probe_session(gcmd)
        probe_num = 0
        while 1:
            self._raise_tool(not probe_num)
            if probe_num >= len(self.probe_points):
                results = probe_session.pull_probed_results()
                done = self._invoke_callback(results)
                if done:
                    break
                # Caller wants a "retry" - restart probing
                probe_num = 0
            self._move_next(probe_num)
            probe_session.run_probe(gcmd)
            probe_num += 1
        probe_session.end_probe_session()
    def _manual_probe_start(self):
        self._raise_tool(not self.manual_results)
        if len(self.manual_results) >= len(self.probe_points):
            done = self._invoke_callback(self.manual_results)
            if done:
                return
            # Caller wants a "retry" - clear results and restart probing
            self.manual_results = []
        self._move_next(len(self.manual_results))
        gcmd = self.gcode.create_gcode_command("", "", {})
        manual_probe.ManualProbeHelper(self.printer, gcmd,
                                       self._manual_probe_finalize)
    def _manual_probe_finalize(self, kin_pos):
        if kin_pos is None:
            return
        self.manual_results.append(kin_pos)
        self._manual_probe_start()

# Helper to obtain a single probe measurement
def run_single_probe(probe, gcmd):
    probe_session = probe.start_probe_session(gcmd)
    probe_session.run_probe(gcmd)
    pos = probe_session.pull_probed_results()[0]
    probe_session.end_probe_session()
    return pos


######################################################################
# Handle [probe] config
######################################################################

# Endstop wrapper that enables probe specific features
class ProbeEndstopWrapper:
    def __init__(self, config):
        self.printer = config.get_printer()
        self._config_section_name = config.get_name() # Store the original config section name
        self.position_endstop = config.getfloat('z_offset')
        self.stow_on_each_sample = config.getboolean(
            'deactivate_on_each_sample', True)
        gcode_macro = self.printer.load_object(config, 'gcode_macro')
        self.activate_gcode = gcode_macro.load_template(
            config, 'activate_gcode', '')
        self.deactivate_gcode = gcode_macro.load_template(
            config, 'deactivate_gcode', '')
        # Create an "endstop" object to handle the probe pin
        ppins = self.printer.lookup_object('pins')
        self.mcu_endstop = ppins.setup_pin('endstop', config.get('pin'))
        # Wrappers
        self.get_mcu = self.mcu_endstop.get_mcu
        self.add_stepper = self.mcu_endstop.add_stepper
        self.get_steppers = self.mcu_endstop.get_steppers
        self.home_start = self.mcu_endstop.home_start
        self.home_wait = self.mcu_endstop.home_wait
        self.query_endstop = self.mcu_endstop.query_endstop
        # multi probes state
        self.multi = 'OFF'

    def get_config_section_name(self):
        return self._config_section_name

    def _raise_probe(self):
        toolhead = self.printer.lookup_object('toolhead')
        start_pos = toolhead.get_position()
        self.deactivate_gcode.run_gcode_from_command()
        if toolhead.get_position()[:3] != start_pos[:3]:
            raise self.printer.command_error(
                "Toolhead moved during probe deactivate_gcode script")
    def _lower_probe(self):
        toolhead = self.printer.lookup_object('toolhead')
        start_pos = toolhead.get_position()
        self.activate_gcode.run_gcode_from_command()
        if toolhead.get_position()[:3] != start_pos[:3]:
            raise self.printer.command_error(
                "Toolhead moved during probe activate_gcode script")
    def multi_probe_begin(self):
        if self.stow_on_each_sample:
            return
        self.multi = 'FIRST'
    def multi_probe_end(self):
        if self.stow_on_each_sample:
            return
        self._raise_probe()
        self.multi = 'OFF'
    def probe_prepare(self, hmove):
        if self.multi == 'OFF' or self.multi == 'FIRST':
            self._lower_probe()
            if self.multi == 'FIRST':
                self.multi = 'ON'
    def probe_finish(self, hmove):
        if self.multi == 'OFF':
            self._raise_probe()
    def get_position_endstop(self):
        return self.position_endstop

# Main external probe interface
class PrinterProbe:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.probe_endstops = {}
        self.probe_offsets_helpers = {}
        self.probe_param_helpers = {}
        self.active_probe_name = None

        probe_prefix = 'probe_tool'
        # Get specific sections for probe_tool*
        tool_probe_configs = config.get_printer().lookup_object(
            'configfile').get_prefix_sections(probe_prefix)
        # Check for the global [probe] section
        legacy_probe_config_section = config.getsection('probe', None)

        if legacy_probe_config_section is not None and tool_probe_configs:
            raise config.error("Cannot use both [probe] and [%s*] sections. "
                               "Please use only [%s*] sections for multiple probes "
                               "or only [probe] for a single probe."
                               % (probe_prefix, probe_prefix))

        if not tool_probe_configs and legacy_probe_config_section is not None:
            # Only legacy [probe] section exists
            probe_name = "default_legacy_probe" 
            self.active_probe_name = probe_name
            cfg_section = legacy_probe_config_section
            self.probe_endstops[probe_name] = ProbeEndstopWrapper(cfg_section)
            self.probe_offsets_helpers[probe_name] = ProbeOffsetsHelper(cfg_section)
            self.probe_param_helpers[probe_name] = ProbeParameterHelper(cfg_section)
            logging.info("Using legacy [probe] configuration as '%s'", probe_name)
        elif tool_probe_configs:
            # New style [probe_tool*] sections exist
            for section_config in tool_probe_configs:
                section_name = section_config.get_name()
                # Extract tool identifier, e.g. "0" from "probe_tool0"
                # or "myprobe" from "probe_toolmyprobe"
                probe_name_suffix = section_name.split(probe_prefix, 1)[1]
                if not probe_name_suffix:
                    # If section is just "probe_tool", assign a default name or error
                    # For now, let's generate a unique-ish name, or log a warning.
                    # This case should ideally be discouraged in documentation.
                    probe_name = section_name + "_unnamed" 
                    logging.warning(
                        "Probe section '%s' does not have a suffix. "
                        "Using generated name: '%s'", section_name, probe_name)
                else:
                    probe_name = probe_name_suffix

                if probe_name in self.probe_endstops:
                    raise config.error("Duplicate probe tool name: %s (from section %s)"
                                       % (probe_name, section_name))
                
                self.probe_endstops[probe_name] = ProbeEndstopWrapper(section_config)
                self.probe_offsets_helpers[probe_name] = ProbeOffsetsHelper(section_config)
                self.probe_param_helpers[probe_name] = ProbeParameterHelper(section_config)
                if self.active_probe_name is None:
                    self.active_probe_name = probe_name # Set first one as active
            if self.active_probe_name is None: # Should not happen if tool_probe_configs is not empty
                 raise config.error("No valid [%s*] sections could be processed." % probe_prefix)
            logging.info("Loaded %d probe tools. Active probe: '%s'",
                         len(self.probe_endstops), self.active_probe_name)
        else:
            # No probe configuration found
            raise config.error("No [probe] or [%s*] configuration sections found."
                               % probe_prefix)

        # Initialize core components with the initially active probe
        self.mcu_probe = self.probe_endstops[self.active_probe_name]
        self.active_probe_offsets_helper = self.probe_offsets_helpers[self.active_probe_name]
        self.active_param_helper = self.probe_param_helpers[self.active_probe_name]

        # Initialize helpers. Pass the main 'config' object for global settings,
        # and specific active probe components where needed.
        # ProbeCommandHelper's config is the main [probe] or [probe_tool<name>] that it's logically tied to.
        # Since commands are registered globally, perhaps it's better to pass the main config.
        # For now, using the main config for cmd_helper.
        # It might need a way to access the active probe's name for messages.
        self.cmd_helper = ProbeCommandHelper(config, self, self.mcu_probe.query_endstop)
        
        self.homing_helper = HomingViaProbeHelper(config, self.mcu_probe,
                                                  self.active_param_helper)
        self.probe_session = ProbeSessionHelper(
            config, self.active_param_helper, self.homing_helper.start_probe_session)

        # Register event handlers
        self.printer.register_event_handler("klippy:connect", self._handle_connect)
        self.printer.register_event_handler("toolhead:active_extruder_changed",
                                            self.handle_active_extruder_changed)

    def _handle_connect(self):
        # Set initial probe based on current extruder
        try:
            toolhead = self.printer.lookup_object('toolhead', None)
            if toolhead:
                active_extruder = toolhead.get_extruder()
                if active_extruder:
                    active_extruder_name = active_extruder.get_name()
                    logging.info(
                        "Initial active extruder: %s. Attempting to set corresponding probe.",
                        active_extruder_name)
                    self.handle_active_extruder_changed(active_extruder_name)
                else:
                    logging.info("No active extruder found at connect time.")
            else:
                logging.info("Toolhead not found at connect time for initial probe sync.")
        except Exception as e:
            logging.warning("Error during initial probe sync with extruder: %s", str(e))


    def handle_active_extruder_changed(self, active_extruder_name):
        if not active_extruder_name:
            logging.info("Active extruder name is None or empty. No probe change.")
            return

        probe_suffix = None
        if active_extruder_name == 'extruder':
            probe_suffix = '0'
        elif active_extruder_name.startswith('extruder') and \
             active_extruder_name[len('extruder):].isdigit():
            probe_suffix = active_extruder_name[len('extruder'):]
        else:
            # Fallback for custom extruder names like "extruder_left", map to "left"
            # This is a simple heuristic; more complex mapping might be needed for arbitrary names.
            # Or, users could name their probes "probe_tool<extruder_name_suffix>"
            if '_' in active_extruder_name: # e.g. extruder_hotend1 -> hotend1
                probe_suffix = active_extruder_name.split('_',1)[1] 
            else: # If no "extruder" prefix and no underscore, maybe the name itself is the suffix
                probe_suffix = active_extruder_name


        if probe_suffix is not None:
            if probe_suffix in self.probe_endstops:
                logging.info("Extruder changed to '%s'. Setting active probe to tool '%s'.",
                             active_extruder_name, probe_suffix)
                try:
                    self._set_active_probe(probe_suffix)
                except Exception as e: # Catch potential errors from _set_active_probe
                    logging.error("Error setting active probe to '%s': %s", probe_suffix, e)
            else:
                # Check if legacy probe "default_legacy_probe" should be activated
                # e.g. if extruder is "extruder" (tool 0) and "probe_tool0" doesn't exist
                # but "default_legacy_probe" does.
                if probe_suffix == '0' and "default_legacy_probe" in self.probe_endstops and \
                   not ('0' in self.probe_endstops and self.probe_endstops['0'] is not self.probe_endstops["default_legacy_probe"]):
                    logging.info("Extruder changed to '%s'. No specific probe tool '%s' found. "
                                 "Activating 'default_legacy_probe'.",
                                 active_extruder_name, probe_suffix)
                    try:
                        self._set_active_probe("default_legacy_probe")
                    except Exception as e:
                         logging.error("Error setting active probe to 'default_legacy_probe': %s", e)
                else:
                    logging.warning("Extruder changed to '%s', but no corresponding probe tool '%s' "
                                    "or applicable default_legacy_probe found.",
                                    active_extruder_name, probe_suffix)
        else:
            logging.warning("Could not determine probe suffix for extruder '%s'. No probe change.",
                            active_extruder_name)

    def _set_active_probe(self, probe_name):
        if probe_name not in self.probe_endstops:
            raise self.printer.command_error( # Use command_error for gcode callable context
                "Probe tool '%s' not configured." % probe_name)
        
        if self.active_probe_name == probe_name:
            # No change needed
            logging.info("Probe tool '%s' is already active.", probe_name)
            return

        logging.info("Setting active probe to '%s'", probe_name)
        self.active_probe_name = probe_name
        
        # Update core components
        self.mcu_probe = self.probe_endstops[self.active_probe_name]
        self.active_probe_offsets_helper = self.probe_offsets_helpers[self.active_probe_name]
        self.active_param_helper = self.probe_param_helpers[self.active_probe_name]

        # Update helper objects that depend on the active probe
        # 1. Update ProbeCommandHelper's query_endstop source
        #    Directly updating the attribute. This assumes query_endstop is accessed as self.query_endstop
        #    within ProbeCommandHelper's methods.
        if self.cmd_helper is not None:
             self.cmd_helper.query_endstop = self.mcu_probe.query_endstop
        
        # 2. Re-initialize HomingViaProbeHelper
        #    This helper registers event handlers. Klipper's event system typically
        #    replaces handlers if the same method is registered for an event.
        #    However, to be safe, if HomingViaProbeHelper had a cleanup method,
        #    it would be called here. For now, direct re-initialization.
        #    We need the main config object that was used to init PrinterProbe.
        main_config = self.printer.lookup_object('configfile').get_config()
        self.homing_helper = HomingViaProbeHelper(main_config, self.mcu_probe,
                                                  self.active_param_helper)
        
        # 3. Re-initialize ProbeSessionHelper
        #    This depends on the active_param_helper and the new homing_helper's
        #    start_probe_session method.
        self.probe_session = ProbeSessionHelper(
            main_config, self.active_param_helper, self.homing_helper.start_probe_session)
        
        self.printer.send_event("probe:active_probe_changed", self.active_probe_name)
        logging.info("Active probe changed to '%s'", self.active_probe_name)

    def get_probe_params(self, gcmd=None):
        return self.active_param_helper.get_probe_params(gcmd)

    def get_offsets(self):
        return self.active_probe_offsets_helper.get_offsets()

    def get_status(self, eventtime):
        # cmd_helper.name might need to be updated if it's tied to a specific section name
        # For now, it uses the name of the config section it was initialized with.
        # This might be okay if cmd_helper is mostly generic.
        return self.cmd_helper.get_status(eventtime)

    def start_probe_session(self, gcmd):
        # Ensure probe_session uses the currently active param_helper and homing_helper
        # This implies that if the active probe changes, probe_session might need to be
        # reconfigured or recreated if its internal state is tied to a specific probe's params.
        # For now, assuming it uses the active_param_helper passed at its init.
        # If active_param_helper is a reference that PrinterProbe updates, it should be fine.
        return self.probe_session.start_probe_session(gcmd)

def load_config(config):
    return PrinterProbe(config)
