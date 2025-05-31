# Klipper module for PrusaSlicer Integration Support
#
# Copyright (C) 2023 <Your Name>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class SlicerIntegration:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')

        # Internal state
        self.current_slicing_job = {
            "model_name": None,
            "expected_gcode_filename": None,
            "status": "idle", # idle, selected, slicing, successful, failed
            "last_message": ""
        }

        # Register G-Code commands
        self.gcode.register_command(
            "SLICER_SELECT_MODEL", self.cmd_SLICER_SELECT_MODEL,
            desc=self.cmd_SLICER_SELECT_MODEL_help
        )
        self.gcode.register_command(
            "SLICER_JOB_STATUS", self.cmd_SLICER_JOB_STATUS,
            desc=self.cmd_SLICER_JOB_STATUS_help
        )

        # Register for status queries (optional, if KlipperScreen needs to query this via Moonraker)
        self.printer.add_object("slicer_job", self)


    cmd_SLICER_SELECT_MODEL_help = "Selects a model and notes the expected G-code filename from a remote slicing job."
    def cmd_SLICER_SELECT_MODEL(self, gcmd):
        model_name = gcmd.get_str('NAME', None)
        expected_gcode_filename = gcmd.get_str('EXPECTED_GCODE_FILENAME', None)

        if model_name is None or expected_gcode_filename is None:
            raise gcmd.error("SLICER_SELECT_MODEL requires NAME and EXPECTED_GCODE_FILENAME parameters.")

        self.current_slicing_job["model_name"] = model_name
        self.current_slicing_job["expected_gcode_filename"] = expected_gcode_filename
        self.current_slicing_job["status"] = "selected"
        self.current_slicing_job["last_message"] = f"Model '{model_name}' selected, expecting G-code file '{expected_gcode_filename}'."

        logging.info(f"[Slicer Integration] Model selected: {model_name}, expecting: {expected_gcode_filename}")
        gcmd.respond_info(f"Model '{model_name}' selected. Expected G-code: '{expected_gcode_filename}'.")

    cmd_SLICER_JOB_STATUS_help = "Updates the status of the current remote slicing job."
    def cmd_SLICER_JOB_STATUS(self, gcmd):
        status = gcmd.get_str('STATUS').upper() # SLICING_STARTED, SLICING_SUCCESSFUL, SLICING_FAILED
        filename = gcmd.get_str('FILENAME', None)
        message = gcmd.get_str('MESSAGE', '')

        valid_statuses = ["SLICING_STARTED", "SLICING_SUCCESSFUL", "SLICING_FAILED"]
        if status not in valid_statuses:
            raise gcmd.error(f"Invalid STATUS provided. Must be one of {valid_statuses}.")

        if status == "SLICING_SUCCESSFUL" and filename is None:
            raise gcmd.error("FILENAME must be provided when STATUS is SLICING_SUCCESSFUL.")

        if status == "SLICING_SUCCESSFUL" and filename != self.current_slicing_job["expected_gcode_filename"]:
            logging.warning(
                f"[Slicer Integration] SLICING_SUCCESSFUL filename '{filename}' "
                f"does not match expected '{self.current_slicing_job['expected_gcode_filename']}'. Using provided filename."
            )
            # Optionally, update the expected filename if this is acceptable behavior
            # self.current_slicing_job["expected_gcode_filename"] = filename

        self.current_slicing_job["status"] = status.lower()
        self.current_slicing_job["last_message"] = message if message else f"Job status updated to {status}."

        log_message = f"[Slicer Integration] Job status: {status}"
        if filename:
            log_message += f", Filename: {filename}"
        if message:
            log_message += f", Message: {message}"
        logging.info(log_message)

        response_message = f"Slicing job status updated to {status}."
        if status == "SLICING_SUCCESSFUL":
            self.current_slicing_job["last_message"] = f"Successfully sliced {self.current_slicing_job['model_name']} to {filename}. {message}".strip()
            response_message += f" Output G-code: '{filename}' is ready."
            # Klipper does not automatically start printing. Moonraker/KlipperScreen will issue a separate print command.
        elif status == "SLICING_FAILED":
            self.current_slicing_job["last_message"] = f"Failed to slice {self.current_slicing_job['model_name']}. Error: {message}".strip()
            response_message += f" Error: {message}"
            # Clear info about the failed job if needed, or keep for inspection
            # self.current_slicing_job["model_name"] = None
            # self.current_slicing_job["expected_gcode_filename"] = None

        gcmd.respond_info(response_message)

    def get_status(self, eventtime):
        # This method allows Moonraker (via webhooks.QueryStatusHelper) to query the status of the slicing job
        return self.current_slicing_job

def load_config(config):
    return SlicerIntegration(config)

def load_config_prefix(config):
    return SlicerIntegration(config)
