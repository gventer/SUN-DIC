from PyQt6 import QtGui
from PyQt6.QtGui import QValidator, QIntValidator, QDoubleValidator


class OddNumberValidator(QValidator):
    """ Custom OddNumberValidator - with upper and lower bounds that are optional
        but inclusive.
    """

    # ---------------------------------------------------------------------------
    def __init__(self, minVal=None, maxVal=None):
        """ Validator to ensure input is an odd integer within optional min and max bounds.
        Args:
            minVal (int, optional): Minimum acceptable value. Defaults to None.
            maxVal (int, optional): Maximum acceptable value. Defaults to None.
        """
        super().__init__()

        self.minVal = minVal
        self.maxVal = maxVal

    # ---------------------------------------------------------------------------
    def validate(self, input_str, pos):
        """ Validate the input string.
        Args:
            input_str (str): The input string to validate.
            pos (int): The current cursor position.
        Returns:
            tuple: (state, input_str, pos) where state is one of QValidator.State.Acceptable,
                   QValidator.State.Intermediate, or QValidator.State.Invalid.
        """
        if input_str == "":
            return (QValidator.State.Intermediate, input_str, pos)

        if not input_str.isdigit():
            return (QValidator.State.Invalid, input_str, pos)

        if input_str.startswith('0'):
            return (QValidator.State.Intermediate, input_str, pos)

        try:
            value = int(input_str)
            # Odd check
            if value % 2 == 1:
                if (self.minVal is None or value >= self.minVal) and \
                   (self.maxVal is None or value <= self.maxVal):
                    return (QValidator.State.Acceptable, input_str, pos)
            return (QValidator.State.Intermediate, input_str, pos)
        except ValueError:
            return (QValidator.State.Invalid, input_str, pos)

    # ----------------------------------------------------------------------------
    def fixup(self, input_str):
        """ Attempt to fix the input string to a valid odd integer within bounds.
        Args:
            input_str (str): The input string to fix.
        Returns:
            str: A fixed valid odd integer string.
        """

        try:
            value = int(input_str)
            if self.minVal is not None and value < self.minVal:
                value = self.minVal
            elif self.maxVal is not None and value > self.maxVal:
                value = self.maxVal
            elif value != 0 and value % 2 == 0:
                return str(value + 1)
        except ValueError:
            if self.minVal is not None:
                value = str(self.minVal)
            else:
                value = str(0)
        return str(value)


class ClampingIntValidator(QIntValidator):
    """ Custom ClampingIntValidator - ensures input is an integer within specified bounds
    """

    # ----------------------------------------------------------------------------
    def fixup(self, input_str):
        """ Attempt to fix the input string to a valid integer within bounds.
        Args:
            input_str (str): The input string to fix.
        Returns:
            str: A fixed valid integer string.
        """

        # Attempt to convert input to int
        try:
            value = int(input_str)
        except ValueError:
            # Default to the minimum if not a valid integer
            return str(self.bottom())

        # Clamp to bounds
        if value < self.bottom():
            return str(self.bottom())
        elif value > self.top():
            return str(self.top())
        return str(value)


class ClampingDblValidator(QtGui.QValidator):
    """ Custom DoubleValidator - with upper and lower bounds that are optional
        but exclusive.
    """

    # ---------------------------------------------------------------------------
    def __init__(self, minVal=None, maxVal=None):
        """ Validator to ensure input is a float within optional min and max bounds.
        Args:
            minVal (float, optional): Minimum acceptable value. Defaults to None.
            maxVal (float, optional): Maximum acceptable value. Defaults to None.
        """
        super().__init__()

        self.minVal = minVal
        self.maxVal = maxVal

    # ---------------------------------------------------------------------------
    def validate(self, input_str, pos):
        """ Validate the input string.
        Args:
            input_str (str): The input string to validate.
            pos (int): The current cursor position.
        Returns:
            tuple: (state, input_str, pos) where state is one of QValidator.State.Acceptable,
                   QValidator.State.Intermediate, or QValidator.State.Invalid.
        """
        if input_str == "":
            return (QValidator.State.Intermediate, input_str, pos)

        # Ok to start with a decimal point
        if input_str == ".":
            return (QValidator.State.Intermediate, input_str, pos)

        # Do not allow non-numeric characters except for the decimal point
        if not all(c.isdigit() or c == '.' for c in input_str):
            return (QValidator.State.Invalid, input_str, pos)

        try:
            value = float(input_str)
            if (self.minVal is None or value > self.minVal) and \
                    (self.maxVal is None or value < self.maxVal):
                return (QValidator.State.Acceptable, input_str, pos)
            return (QValidator.State.Intermediate, input_str, pos)
        except ValueError:
            return (QtGui.QValidator.State.Invalid, input_str, pos)

    # ---------------------------------------------------------------------------
    def fixup(self, input_str):
        """ Attempt to fix the input string to a valid float within bounds.
        Args:
            input_str (str): The input string to fix.
        Returns:
            str: A fixed valid float string.
        """
        epsilon = 1e-12
        try:
            value = float(input_str)
            if self.minVal is not None and value <= self.minVal:
                value = self.minVal+epsilon
            elif self.maxVal is not None and value >= self.maxVal:
                value = self.maxVal-epsilon
        except ValueError:
            if self.minVal is not None:
                value = str(self.minVal+epsilon)
            else:
                value = str(epsilon)

        return str(value)


class IntListValidator(QIntValidator):
    """ Custom IntListValidator - ensures input is an integer from a list of acceptable values
    """

    # ---------------------------------------------------------------------------
    def __init__(self, values=[]):
        """ Validator to ensure input is an integer from a list of acceptable values.
        Args:
            values (list): List of acceptable integer values.
        """
        super().__init__()

        self._values = values

    # ---------------------------------------------------------------------------
    def validate(self, input_str, pos):
        """ Validate the input string.
        Args:
            input_str (str): The input string to validate.
            pos (int): The current cursor position.
        Returns:
            tuple: (state, input_str, pos) where state is one of QValidator.State.Acceptable,
                   QValidator.State.Intermediate, or QValidator.State.Invalid.
        """

        state, _, _ = super().validate(input_str, pos)

        try:
            value = int(input_str)
            if value in self._values:
                return (QValidator.State.Acceptable, input_str, pos)
            return (QValidator.State.Intermediate, input_str, pos)
        except ValueError:
            return (QValidator.State.Invalid, input_str, pos)

    # ---------------------------------------------------------------------------
    def fixup(self, input_str):
        """ Attempt to fix the input string to a valid integer within the list of acceptable values.
        Args:
            input_str (str): The input string to fix.
        Returns:
            str: A fixed valid integer string.
        """
        try:
            value = int(input_str)
            closest = min(self._values, key=lambda x: abs(x - value))
            value = closest
        except ValueError:
            if self._values is not None and len(self._values) > 0:
                value = str(self._values[0])
            else:
                value = str(1)
        return str(value)
