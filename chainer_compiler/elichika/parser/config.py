import sys
import logging

# whether it shows warnings while compiling
show_warnings = True

# whether float64 isn't regarded as float32
float_restrict = False

# registerd module are ignored while parsing
disabled_modules = set()
disabled_modules.add(logging)
disabled_modules.add(sys)
