# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'resultsSumWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1114, 615)
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 40, 901, 375))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(4)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout.setContentsMargins(0, 0, -1, -1)
        self.gridLayout.setHorizontalSpacing(4)
        self.gridLayout.setVerticalSpacing(5)
        self.gridLayout.setObjectName("gridLayout")
        self.incStrainsIn = QtWidgets.QCheckBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.incStrainsIn.sizePolicy().hasHeightForWidth())
        self.incStrainsIn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.incStrainsIn.setFont(font)
        self.incStrainsIn.setText("")
        self.incStrainsIn.setObjectName("incStrainsIn")
        self.gridLayout.addWidget(self.incStrainsIn, 2, 3, 1, 1, QtCore.Qt.AlignLeft)
        self.incDispIn = QtWidgets.QCheckBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.incDispIn.sizePolicy().hasHeightForWidth())
        self.incDispIn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.incDispIn.setFont(font)
        self.incDispIn.setText("")
        self.incDispIn.setObjectName("incDispIn")
        self.gridLayout.addWidget(self.incDispIn, 2, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.incDispLab = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.incDispLab.sizePolicy().hasHeightForWidth())
        self.incDispLab.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.incDispLab.setFont(font)
        self.incDispLab.setObjectName("incDispLab")
        self.gridLayout.addWidget(self.incDispLab, 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.imgPairLab = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imgPairLab.sizePolicy().hasHeightForWidth())
        self.imgPairLab.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.imgPairLab.setFont(font)
        self.imgPairLab.setObjectName("imgPairLab")
        self.gridLayout.addWidget(self.imgPairLab, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.incStrainsLab = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.incStrainsLab.sizePolicy().hasHeightForWidth())
        self.incStrainsLab.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.incStrainsLab.setFont(font)
        self.incStrainsLab.setObjectName("incStrainsLab")
        self.gridLayout.addWidget(self.incStrainsLab, 2, 2, 1, 1, QtCore.Qt.AlignLeft)
        self.imgPairIn = QtWidgets.QComboBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imgPairIn.sizePolicy().hasHeightForWidth())
        self.imgPairIn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.imgPairIn.setFont(font)
        self.imgPairIn.setObjectName("imgPairIn")
        self.gridLayout.addWidget(self.imgPairIn, 0, 1, 1, 1)
        self.removeNanLab = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.removeNanLab.sizePolicy().hasHeightForWidth())
        self.removeNanLab.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.removeNanLab.setFont(font)
        self.removeNanLab.setObjectName("removeNanLab")
        self.gridLayout.addWidget(self.removeNanLab, 0, 2, 1, 1)
        self.removeNanIn = QtWidgets.QCheckBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.removeNanIn.sizePolicy().hasHeightForWidth())
        self.removeNanIn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.removeNanIn.setFont(font)
        self.removeNanIn.setText("")
        self.removeNanIn.setObjectName("removeNanIn")
        self.gridLayout.addWidget(self.removeNanIn, 0, 3, 1, 1)
        self.smoothOrderIn = QtWidgets.QLineEdit(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.smoothOrderIn.sizePolicy().hasHeightForWidth())
        self.smoothOrderIn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.smoothOrderIn.setFont(font)
        self.smoothOrderIn.setObjectName("smoothOrderIn")
        self.gridLayout.addWidget(self.smoothOrderIn, 1, 3, 1, 1)
        self.smoothOrderLab = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.smoothOrderLab.sizePolicy().hasHeightForWidth())
        self.smoothOrderLab.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.smoothOrderLab.setFont(font)
        self.smoothOrderLab.setObjectName("smoothOrderLab")
        self.gridLayout.addWidget(self.smoothOrderLab, 1, 2, 1, 1)
        self.smoothWindowLab = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.smoothWindowLab.sizePolicy().hasHeightForWidth())
        self.smoothWindowLab.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.smoothWindowLab.setFont(font)
        self.smoothWindowLab.setObjectName("smoothWindowLab")
        self.gridLayout.addWidget(self.smoothWindowLab, 1, 0, 1, 1)
        self.smoothWindowIn = QtWidgets.QLineEdit(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.smoothWindowIn.sizePolicy().hasHeightForWidth())
        self.smoothWindowIn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.smoothWindowIn.setFont(font)
        self.smoothWindowIn.setText("")
        self.smoothWindowIn.setObjectName("smoothWindowIn")
        self.gridLayout.addWidget(self.smoothWindowIn, 1, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.writeDataBut = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.writeDataBut.sizePolicy().hasHeightForWidth())
        self.writeDataBut.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        self.writeDataBut.setFont(font)
        self.writeDataBut.setObjectName("writeDataBut")
        self.verticalLayout_2.addWidget(self.writeDataBut, 0, QtCore.Qt.AlignHCenter)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.incDispLab.setText(_translate("Form", "Include Displacements:"))
        self.imgPairLab.setText(_translate("Form", "Image Pair:"))
        self.incStrainsLab.setText(_translate("Form", "Include Strains:"))
        self.removeNanLab.setText(_translate("Form", "Remove NAN\'s:"))
        self.smoothOrderLab.setText(_translate("Form", "Smoothing Order:"))
        self.smoothWindowLab.setText(_translate("Form", "Smoothing Window:"))
        self.writeDataBut.setText(_translate("Form", "Write Data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
