QT = core
CONFIG += c++17 console
TARGET = analysis_app

QT -= gui widgets charts network

SOURCES += src/main.cpp \
           src/auto_kmeans.cpp \
           src/dbscan.cpp \
           src/kde.cpp \
           src/analysis_utils.cpp \
           src/data_analyzer.cpp \
           src/em.cpp \
           src/metric.cpp

HEADERS += src/auto_kmeans.hpp \
           src/dbscan.hpp \
           src/kde.hpp \
           src/analysis_utils.hpp \
           src/data_analyzer.hpp \
           src/em.hpp \
           src/metric.hpp

QMAKE_CXXFLAGS += -std=c++17 -fPIC
