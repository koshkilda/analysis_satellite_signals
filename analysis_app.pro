QT = core
CONFIG += c++17 console
TARGET = analysis_app

QT -= gui widgets charts network

SOURCES += main.cpp \
           auto_kmeans.cpp \
           dbscan.cpp \
           kde.cpp \
           analysis_utils.cpp \
           data_analyzer.cpp \
           em.cpp \
           metric.cpp

HEADERS += auto_kmeans.hpp \
           dbscan.hpp \
           kde.hpp \
           analysis_utils.hpp \
           data_analyzer.hpp \
           em.hpp \
           metric.hpp

QMAKE_CXXFLAGS += -std=c++17 -fPIC