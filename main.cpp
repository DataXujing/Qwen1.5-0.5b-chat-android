#include "widget.h"

#include <QApplication>
//#include <QVBoxLayout>
#include "llm.hpp"
#include <QDebug>
#include <QFile>
#include <QDir>

std::unique_ptr<Llm> llm(nullptr);


bool copyDirectoryFiles(const QString &fromDir, const QString &toDir, bool coverFileIfExist)
{
    QDir sourceDir(fromDir);
    QDir targetDir(toDir);
    if(!targetDir.exists()){    /**< 如果目标目录不存在，则进行创建 */
        if(!targetDir.mkdir(targetDir.absolutePath()))
            return false;
    }

    QFileInfoList fileInfoList = sourceDir.entryInfoList();
    foreach(QFileInfo fileInfo, fileInfoList){
        if(fileInfo.fileName() == "." || fileInfo.fileName() == "..")
            continue;

        if(fileInfo.isDir()){    /**< 当为目录时，递归的进行copy */
            if(!copyDirectoryFiles(fileInfo.filePath(),
                targetDir.filePath(fileInfo.fileName()),
                coverFileIfExist))
                return false;
        }
        else{            /**< 当允许覆盖操作时，将旧文件进行删除操作 */
            if(coverFileIfExist && targetDir.exists(fileInfo.fileName())){
                targetDir.remove(fileInfo.fileName());
            }

            /// 进行文件copy
            if(!QFile::copy(fileInfo.filePath(),
                targetDir.filePath(fileInfo.fileName()))){
                    return false;
            }
        }
    }
    return true;
}


int main(int argc, char *argv[])
{
    //    qputenv("QT_IM_MODULE",QByteArray("qtvirtualkeyboard"));
    QApplication a(argc, argv);
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    //模型加载放在main函数
    //    std::string model_dir ="assets:/models";
    copyDirectoryFiles("assets:/models","models", false);
    if (!llm.get()) {
        llm.reset(Llm::createLLM("models","qwen1.5-0.5b-chat"));
        llm->load("models");
        qDebug() << "---------------------------------------->>>>: Qwen1.5-0.5b模型加载完成";


    }

    Widget w;
    //    QVBoxLayout layout;
    //    w.setLayout(&layout);
    //    w.showFullScreen();
    w.show();
    return a.exec();
}
