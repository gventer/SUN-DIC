from PyQt5 import QtCore, QtGui
import typing

class OddNumberValidator(QtGui.QValidator):
    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)

    def validate(self, input_str: str, pos: int) -> typing.Tuple[QtGui.QValidator.State, str, int]:
        if ' ' in input_str:
            return (QtGui.QValidator.Intermediate, input_str, pos)
        
        if input_str.isdigit():
            try:
                value = int(input_str)
                if value % 2 != 0:
                    if 1 <= value:
                        return (QtGui.QValidator.Acceptable, input_str, pos)
                    else:
                        return (QtGui.QValidator.Intermediate, input_str, pos)
                else:
                    return (QtGui.QValidator.Intermediate, input_str, pos)
            except ValueError:
                return (QtGui.QValidator.Intermediate, input_str, pos)
        else:
            return (QtGui.QValidator.Intermediate, input_str, pos)

    def fixup(self, input_str: str) -> str:
        input_str = ''.join(filter(str.isdigit, input_str))
        
        try:
            value = int(input_str)
            if value < 1:
                return '1'
            elif value % 2 == 0:
                return str(value + 1)
        except ValueError:
            return '1'
        return '5'
    
class PositiveIntValidator(QtGui.QValidator):
    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)

    def validate(self, input_str: str, pos: int) -> typing.Tuple[QtGui.QValidator.State, str, int]:
        if ' ' in input_str:
            return (QtGui.QValidator.Intermediate, input_str, pos)
        
        if input_str.isdigit():
            try:
                value = int(input_str)
                if value > 0:
                    return (QtGui.QValidator.Acceptable, input_str, pos)
                else:
                    return (QtGui.QValidator.Intermediate, input_str, pos)
            except ValueError:
                return (QtGui.QValidator.Intermediate, input_str, pos)
        else:
            return (QtGui.QValidator.Intermediate, input_str, pos)
        
    def fixup(self, input_str: str) -> str:
        input_str = input_str.replace(' ', '')
        
        try:
            value = int(input_str)
            if value <= 0:
                return '1'
        except ValueError:
            return '1'
        
        return input_str
    
class PositiveDoubleValidator(QtGui.QValidator):
    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)

    def validate(self, input_str: str, pos: int) -> typing.Tuple[QtGui.QValidator.State, str, int]:
        if ' ' in input_str:
            return (QtGui.QValidator.Intermediate, input_str, pos)
        
        try:
            value = float(input_str)
            if value > 0:
                return (QtGui.QValidator.Acceptable, input_str, pos)
            else:
                return (QtGui.QValidator.Intermediate, input_str, pos)
        except ValueError:
            return (QtGui.QValidator.Intermediate, input_str, pos)

    def fixup(self, input_str: str) -> str:
        input_str = input_str.replace(' ', '')
        
        try:
            value = float(input_str)
            if value <= 0:
                return '0.0001'
        except ValueError:
            return '0.0001'
        
        return input_str
    
class OddNumberZeroValidator(QtGui.QValidator):
    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)

    def validate(self, input_str: str, pos: int) -> typing.Tuple[QtGui.QValidator.State, str, int]:
        if ' ' in input_str:
            return (QtGui.QValidator.Intermediate, input_str, pos)
        
        if input_str.isdigit():
            try:
                value = int(input_str)
                if value % 2 != 0:
                    if 0 <= value <= 10000:
                        return (QtGui.QValidator.Acceptable, input_str, pos)
                    else:
                        return (QtGui.QValidator.Intermediate, input_str, pos)
                else:
                    return (QtGui.QValidator.Intermediate, input_str, pos)
            except ValueError:
                return (QtGui.QValidator.Intermediate, input_str, pos)
        else:
            return (QtGui.QValidator.Intermediate, input_str, pos)

    def fixup(self, input_str: str) -> str:
        input_str = ''.join(filter(str.isdigit, input_str))
        
        try:
            value = int(input_str)
            if value < 0:
                return '0'
            elif value % 2 == 0 and value != 0:
                return str(value + 1)
        except ValueError:
            return '3'
        return '0'
    
class Int255Validator(QtGui.QValidator):
    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)

    def validate(self, input_str: str, pos: int) -> typing.Tuple[QtGui.QValidator.State, str, int]:
        if ' ' in input_str:
            return (QtGui.QValidator.Intermediate, input_str, pos)
        
        if input_str.isdigit():
            try:
                value = int(input_str)
                if value > 0 and value <= 255:
                    return (QtGui.QValidator.Acceptable, input_str, pos)
                else:
                    return (QtGui.QValidator.Intermediate, input_str, pos)
            except ValueError:
                return (QtGui.QValidator.Intermediate, input_str, pos)
        else:
            return (QtGui.QValidator.Intermediate, input_str, pos)
        
    def fixup(self, input_str: str) -> str:
        input_str = input_str.replace(' ', '')
        
        try:
            value = int(input_str)
            if value <= 0:
                return '1'
            elif value > 255:
                return '255'
        except ValueError:
            return '1'
        
        return input_str