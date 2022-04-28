from enum import Enum, IntEnum, EnumMeta

def check_enum_value(name):
    if isinstance(name, (Enum, IntEnum)):
        name = name.value
    return name

def check_enum_name(name):
    if isinstance(name, (Enum, IntEnum)):
        name = name.name
    return name

def check_enum(arg):
    return check_enum_name(arg), check_enum_value(arg)

def enum_to_dict(enum_holder):
    return {name: value.value for (name, value) in enum_holder.__members__.items()}

def assert_member(enum_holder, member):
    assert member in enum_holder.__members__, (
        "Invalid Enum member. Member must be a valid enum for '{}'. \n"
        "Received: '{}'. Options are (member, value):\n"
        "{}".format(enum_holder.__name__, member, enum_to_dict(enum_holder)))

def assert_value(enum_holder, value):
    assert value in enum_holder.__members__.values(), (
        "Invalid Enum value. Value must be a valid enum for '{}'. \n"
        "Received: '{}'. Options are (member, value):\n"
        "{}".format(enum_holder.__name__, value, enum_to_dict(enum_holder)))