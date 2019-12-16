TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

DESTDIR = $$PWD

SOURCES += \
        main.cpp \
        mnist.cpp \
        rwfile.cpp

FORMS +=

HEADERS += \
	mnist.h \
	rwfile.h
