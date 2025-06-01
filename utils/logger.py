import os
import sys
import logging

no_colorama = False
try:
    from colorama import init as colorama_init, Fore, Back, Style
except ImportError:
    no_colorama = True
# If colorama isn't installed use an ANSI basic replacement
if no_colorama:
    from .ansi import Fore, Back, Style  # noqa: F811
else:
    colorama_init()

NAME = "FLOAT_Optimized"


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    def __init__(self):
        super(logging.Formatter, self).__init__()
        white = Fore.WHITE + Style.BRIGHT
        yellow = Fore.YELLOW + Style.BRIGHT
        red = Fore.RED + Style.BRIGHT
        red_alarm = Fore.RED + Back.WHITE + Style.BRIGHT
        cyan = Fore.CYAN + Style.BRIGHT
        reset = Style.RESET_ALL
        # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
        #          "(%(filename)s:%(lineno)d)"
        format = f"[{NAME} %(levelname)s] %(message)s (%(name)s - %(filename)s:%(lineno)d)"
        format_simple = f"[{NAME}] %(message)s"

        self.FORMATS = {
            logging.DEBUG: cyan + format + reset,
            logging.INFO: white + format_simple + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: red_alarm + format + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Create a new logger
logger = logging.getLogger(NAME)
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

# ######################
# Logger setup
# ######################
# 1. Determine the ComfyUI global log level (influenced by --verbose)
float_logger = logger
comfy_root_logger = logging.getLogger('comfy')
effective_comfy_level = logging.getLogger().getEffectiveLevel()
# 2. Check our custom environment variable for more verbosity
try:
    float_nodes_debug_env = int(os.environ.get("FLOAT_NODES_DEBUG", "0"))
except ValueError:
    float_nodes_debug_env = 0
# 3. Set node's logger level
if float_nodes_debug_env:
    float_logger.setLevel(logging.DEBUG - (float_nodes_debug_env - 1))
    final_level_str = f"DEBUG (due to FLOAT_NODES_DEBUG={float_nodes_debug_env})"
else:
    float_logger.setLevel(effective_comfy_level)
    final_level_str = logging.getLevelName(effective_comfy_level) + " (matching ComfyUI global)"
_initial_setup_logger = logging.getLogger(NAME + ".setup")  # A temporary logger for this message
_initial_setup_logger.debug(f"FLOAT_Nodes logger level set to: {final_level_str}")
