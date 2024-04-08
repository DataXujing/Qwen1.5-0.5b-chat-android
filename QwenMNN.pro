QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    widget.cpp

HEADERS += \
    widget.h

FORMS += \
    widget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    android/AndroidManifest.xml \
    android/build.gradle \
    android/gradle/wrapper/gradle-wrapper.jar \
    android/gradle/wrapper/gradle-wrapper.properties \
    android/gradlew \
    android/gradlew.bat \
    android/res/values/libs.xml

ANDROID_PACKAGE_SOURCE_DIR = $$PWD/android

unix:!macx: LIBS += -L$$PWD/../../mnn-llm-android/libs/ -lMNN
INCLUDEPATH += $$PWD/../../mnn-llm-android/include
DEPENDPATH += $$PWD/../../mnn-llm-android/include
unix:!macx: LIBS += -L$$PWD/../../mnn-llm-android/libs/ -lMNN_Express
unix:!macx: LIBS += -L$$PWD/../../mnn-llm-android/android_build/ -lllm

INCLUDEPATH += $$PWD/../../mnn-llm-android/include
DEPENDPATH += $$PWD/../../mnn-llm-android/include

ANDROID_EXTRA_LIBS = F:/andriod_env/mnn-llm/project/QwenMNN/../../mnn-llm-android/libs/libMNN.so F:/andriod_env/mnn-llm/project/QwenMNN/../../mnn-llm-android/libs/libMNN_Express.so $$PWD/../../mnn-llm-android/android_build/libllm.so


android {
  data.files += qwen_model/block_0.mnn
  data.files += qwen_model/block_1.mnn
  data.files += qwen_model/block_2.mnn
  data.files += qwen_model/block_3.mnn
  data.files += qwen_model/block_4.mnn
  data.files += qwen_model/block_5.mnn
  data.files += qwen_model/block_6.mnn
  data.files += qwen_model/block_7.mnn
  data.files += qwen_model/block_8.mnn
  data.files += qwen_model/block_9.mnn
  data.files += qwen_model/block_10.mnn
  data.files += qwen_model/block_11.mnn
  data.files += qwen_model/block_12.mnn
  data.files += qwen_model/block_13.mnn
  data.files += qwen_model/block_14.mnn
  data.files += qwen_model/block_15.mnn
  data.files += qwen_model/block_16.mnn
  data.files += qwen_model/block_17.mnn
  data.files += qwen_model/block_18.mnn
  data.files += qwen_model/block_19.mnn
  data.files += qwen_model/block_20.mnn
  data.files += qwen_model/block_21.mnn
  data.files += qwen_model/block_22.mnn
  data.files += qwen_model/block_23.mnn
  data.files += qwen_model/embedding.mnn

  data.files += qwen_model/embeddings_bf16.bin
  data.files += qwen_model/lm.mnn
  data.files += qwen_model/tokenizer.txt

  data.path = /assets/models
  INSTALLS += data

}
