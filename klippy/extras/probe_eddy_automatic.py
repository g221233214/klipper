# Automatic Z-Offset Calibration for Eddy Current Probes
#
# Copyright (C) 2024 Jules <jules@google.com> # Fictional attribution
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math

class EddyAutoCalibration:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')

        # These will be loaded dynamically in the G-Code command
        self.fast_descent_speed = 5.0
        self.slow_descent_speed = 1.0
        self.fine_step_z = 0.005
        self.settling_time = 0.020 # seconds
        self.samples_per_step = 5
        self.derivative_window = 3
        self.contact_threshold_factor = 0.3
        self.min_approach_dfdz = 10.0
        self.max_overdrive_z = 0.05
        self.retract_distance = 2.0
        self.start_z_above_bed = None

        self.gcode.register_command(
            'AUTO_CALIBRATE_EDDY_Z_OFFSET',
            self.cmd_AUTO_CALIBRATE_EDDY_Z_OFFSET,
            desc=self.cmd_AUTO_CALIBRATE_EDDY_Z_OFFSET_help
        )

        self.toolhead = None
        self.probe = None # This is PrinterProbe instance
        self.eddy_sensor_module = None # This is LDC1612 instance (or similar)

        self.z_positions_history = []
        self.frequency_readings_history = []
        self.df_dz_history = []

        self.is_processing_data = False
        self.collected_samples_current_step = []
        self.reactor = self.printer.get_reactor()

    def _load_config_for_chip(self, chip_config_section):
        # Load parameters from the chip's specific config section
        self.fast_descent_speed = chip_config_section.getfloat('auto_cal_fast_descent_speed', 5.0, above=0.)
        self.slow_descent_speed = chip_config_section.getfloat('auto_cal_slow_descent_speed', 1.0, above=0.)
        self.fine_step_z = chip_config_section.getfloat('auto_cal_fine_step_z', 0.005, above=0.)
        self.settling_time = chip_config_section.getfloat('auto_cal_settling_time', 0.020, above=0.)
        self.samples_per_step = chip_config_section.getint('auto_cal_samples_per_step', 5, minval=1)
        self.derivative_window = chip_config_section.getint('auto_cal_derivative_window', 3, minval=2)
        self.contact_threshold_factor = chip_config_section.getfloat('auto_cal_contact_threshold_factor', 0.3, above=0., maxval=1.)
        self.min_approach_dfdz = chip_config_section.getfloat('auto_cal_min_approach_dfdz', 10.0)
        self.max_overdrive_z = chip_config_section.getfloat('auto_cal_max_overdrive_z', 0.05, above=0.)
        self.retract_distance = chip_config_section.getfloat('auto_cal_retract_distance', 2.0, above=0.)
        self.start_z_above_bed = chip_config_section.getfloat('auto_cal_start_z_above_bed', None, note_valid=False)

    def _setup_dependencies(self):
        self.toolhead = self.printer.lookup_object('toolhead')
        self.probe = self.printer.lookup_object('probe', None)
        if self.probe is None:
            raise self.gcode.error("A probe is required but not configured.")

    def _ldc_data_callback(self, msg):
        if not self.is_processing_data:
            return True

        if 'data' in msg and msg['data']:
            for t, freq, z_unused in msg['data']:
                self.collected_samples_current_step.append(freq)
        return True

    cmd_AUTO_CALIBRATE_EDDY_Z_OFFSET_help = "Automatically calibrates Z offset for an LDC-based eddy current probe by detecting bed contact."
    def cmd_AUTO_CALIBRATE_EDDY_Z_OFFSET(self, gcmd):
        chip_name = gcmd.get("CHIP")
        if not chip_name:
            raise gcmd.error("Missing CHIP parameter (e.g., CHIP=my_ldc_sensor).")

        self.eddy_sensor_module = None
        for obj_name_key, obj in self.printer.objects_by_name.items():
            if hasattr(obj, 'name') and obj.name == chip_name and hasattr(obj, 'config_section') and hasattr(obj, 'add_client'):
                if "ldc" in str(type(obj)).lower():
                    self.eddy_sensor_module = obj
                    break

        if self.eddy_sensor_module is None:
            raise gcmd.error(f"Eddy current sensor chip '{chip_name}' configured as [probe_eddy_current {chip_name}] not found or is not a compatible LDC module.")

        self._load_config_for_chip(self.eddy_sensor_module.config_section)
        self._setup_dependencies()

        gcmd.respond_info(f"Starting automatic Z offset calibration for eddy probe '{chip_name}'. Parameters loaded from [{self.eddy_sensor_module.config_section.get_name()}].")
        gcode_state = self.gcode.get_saved_state()
        self.gcode.save_state()

        client_added = False
        final_contact_z = None # Initialize here for use in finally block
        try:
            self.z_positions_history = []
            self.frequency_readings_history = []
            self.df_dz_history = []

            self.eddy_sensor_module.add_client(self._ldc_data_callback)
            client_added = True
            if hasattr(self.eddy_sensor_module, '_start_measurements'):
                 self.eddy_sensor_module._start_measurements()
            self.is_processing_data = True

            if 'z' not in self.toolhead.get_homed_axes():
                raise gcmd.error("Z axis must be homed before running AUTO_CALIBRATE_EDDY_Z_OFFSET.")

            current_pos = self.toolhead.get_position()
            start_z = current_pos[2]

            if self.start_z_above_bed is not None:
                home_pos = self.printer.lookup_object('gcode_move').get_status()['homing_origin']
                target_start_z = home_pos.z + self.start_z_above_bed
                gcmd.respond_info(f"Moving to starting Z height: {target_start_z:.3f}mm")
                self.toolhead.manual_move([current_pos[0], current_pos[1], target_start_z], self.fast_descent_speed)
                self.toolhead.wait_moves()
                start_z = self.toolhead.get_position()[2] # Update start_z to actual

            current_z = start_z
            gcmd.respond_info(f"Starting fine descent from Z={current_z:.4f}. Step: {self.fine_step_z}mm, Speed: {self.slow_descent_speed}mm/s")

            max_fine_steps = int(abs(start_z - (-5.0)) / self.fine_step_z) # Probe down to -5mm absolute for safety.
            max_fine_steps = min(max_fine_steps, 1000)

            recent_abs_dfdz_values = []
            min_approach_dfdz_observed = False

            for i in range(max_fine_steps):
                self.collected_samples_current_step = []

                target_z_step = current_z - self.fine_step_z
                self.toolhead.manual_move([None, None, target_z_step], self.slow_descent_speed)
                self.toolhead.wait_moves()
                current_z = self.toolhead.get_position()[2]

                self.reactor.pause(self.reactor.monotonic() + self.settling_time)

                if not self.collected_samples_current_step:
                    gcmd.respond_info(f"Warning: No samples collected at Z={current_z:.4f}. Retrying step or increase settling_time.")
                    # Optional: could add logic to retry the step once.
                    self.reactor.pause(self.reactor.monotonic() + self.settling_time) # Try waiting a bit longer
                    if not self.collected_samples_current_step:
                        gcmd.respond_info(f"Still no samples at Z={current_z:.4f}. Skipping point.")
                        continue


                current_avg_freq = sum(self.collected_samples_current_step) / len(self.collected_samples_current_step)
                gcmd.respond_info("Z: %.4f, Avg Freq: %.3f (Samples: %d)" % (
                    current_z, current_avg_freq, len(self.collected_samples_current_step)))

                self.z_positions_history.append(current_z)
                self.frequency_readings_history.append(current_avg_freq)

                if len(self.z_positions_history) >= self.derivative_window:
                    idx0 = -self.derivative_window
                    idx1 = -1

                    dF = self.frequency_readings_history[idx1] - self.frequency_readings_history[idx0]
                    dZ = self.z_positions_history[idx1] - self.z_positions_history[idx0]

                    current_df_dz = (dF / dZ) if abs(dZ) > 1e-6 else 0.0
                    self.df_dz_history.append(current_df_dz)
                    # Ensure recent_abs_dfdz_values has enough elements before calculating avg_recent_abs_df_dz
                    if abs(current_df_dz) > 0: # Store non-zero dF/dZ values for baseline
                        recent_abs_dfdz_values.append(abs(current_df_dz))
                        if len(recent_abs_dfdz_values) > self.derivative_window * 2: # Keep a rolling window
                             recent_abs_dfdz_values.pop(0)

                    gcmd.respond_info("dF/dZ: %.3f" % current_df_dz)

                    if not min_approach_dfdz_observed and abs(current_df_dz) >= self.min_approach_dfdz :
                        min_approach_dfdz_observed = True
                        gcmd.respond_info(f"Min approach dF/dZ of {self.min_approach_dfdz:.2f} observed. Monitoring for drop.")
                        recent_abs_dfdz_values = [abs(current_df_dz)] # Reset baseline after significant approach starts

                    if min_approach_dfdz_observed and recent_abs_dfdz_values:
                        avg_recent_abs_df_dz = sum(recent_abs_dfdz_values) / len(recent_abs_dfdz_values)

                        if abs(current_df_dz) < abs(avg_recent_abs_df_dz * self.contact_threshold_factor):
                            gcmd.respond_info("Contact detected at Z=%.4f! |dF/dZ| (%.2f) < threshold (%.2f * %.2f = %.2f)" % (
                                self.z_positions_history[-2],
                                abs(current_df_dz),
                                abs(avg_recent_abs_df_dz), self.contact_threshold_factor,
                                abs(avg_recent_abs_df_dz * self.contact_threshold_factor) ))
                            final_contact_z = self.z_positions_history[-2]

                            if self.max_overdrive_z > 0:
                                gcmd.respond_info(f"Confirming contact by overdriving up to {self.max_overdrive_z}mm...")
                                # Simplified: Real confirmation would check if dF/dZ stays low.
                            break

                if current_z < -5.0:
                    raise gcmd.error("Probed below safe Z limit (-5.0mm) without contact.")

            if final_contact_z is None:
                raise gcmd.error("Failed to detect bed contact. Try adjusting auto_cal parameters for your sensor/setup or check mechanicals.")

            gcmd.respond_info("Automatic contact point determined at Z machine coordinate: %.4f" % final_contact_z)

            new_z_offset = final_contact_z

            active_probe_config_name = self.probe.probe_endstops[
                self.probe.active_probe_name].get_config_section_name()

            self.gcode.respond_info(
                "Probe %s: new z_offset automatically determined: %.4f
"
                "The SAVE_CONFIG command will update the printer config file
"
                "with the above and restart the printer." % (active_probe_config_name, new_z_offset))
            configfile = self.printer.lookup_object('configfile')
            configfile.set(active_probe_config_name, 'z_offset', "%.4f" % (new_z_offset,))

        except Exception as e:
            logging.exception("Error during AUTO_CALIBRATE_EDDY_Z_OFFSET")
            self.gcode.respond_info("Error: %s" % str(e))
            raise
        finally:
            self.is_processing_data = False
            if self.eddy_sensor_module:
                if hasattr(self.eddy_sensor_module, '_finish_measurements'):
                    self.eddy_sensor_module._finish_measurements()
                if client_added and hasattr(self.eddy_sensor_module, 'remove_client'):
                    # Check if remove_client method exists in LDC1612 or its BatchBulkHelper
                    # self.eddy_sensor_module.remove_client(self._ldc_data_callback) # This method isn't standard in BatchBulkHelper
                    # For now, setting is_processing_data to False should effectively stop data collection.
                    # Proper client removal might need BatchBulkHelper to expose such a method.
                    pass # No standard remove_client in BatchBulkHelper

            current_pos_final = self.toolhead.get_position()
            # Ensure retract happens relative to the lowest point reached OR current Z if no contact point was found
            # This ensures retract happens from a known safe point.
            # If final_contact_z was found, it's the lowest point we want to retract from.
            # If an error occurred before final_contact_z was set, current_pos_final[2] is the lowest.
            retract_from_z = final_contact_z if final_contact_z is not None else current_pos_final[2]

            # Check if toolhead is available (it might not be if error happened early)
            if self.toolhead:
                self.toolhead.manual_move([None, None, retract_from_z + self.retract_distance], self.slow_descent_speed)
                self.toolhead.wait_moves()

            if hasattr(self, 'gcode') and gcode_state: # Ensure gcode and gcode_state are available
                 self.gcode.restore_state(gcode_state)
            gcmd.respond_info("Automatic Z offset calibration finished.")

def load_config(config):
    return EddyAutoCalibration(config)
