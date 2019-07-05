import sys
import logging

show_warnings = True
float_restrict = False


# registerd module are ignored while parsing
disabled_modules = set()
disabled_modules.add(logging)
disabled_modules.add(sys)
