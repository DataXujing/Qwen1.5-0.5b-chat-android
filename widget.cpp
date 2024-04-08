#include "widget.h"
#include "ui_widget.h"
#include <QDebug>
//#include <QTextDocumentFragment>
#include "llm.hpp"
#include <string>
#include <sstream>
#include <thread>

extern std::unique_ptr<Llm> llm;
static std::stringstream response_buffer;

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    ui->textEdit->setReadOnly(true);


}

Widget::~Widget()
{
    delete ui;
}


void Widget::on_pushButton_clicked()
{


    qDebug() << "这是button 被click" ;
    QString prompt = ui->textEdit_2->toPlainText();
    ui->textEdit_2->clear();
    //    QTextDocumentFragment fragment;
    //    fragment = QTextDocumentFragment::fromHtml("<img src='F:/andriod_env/mnn-llm/project/QwenMNN/pic/qwen.png' width='10%'>");
    //    ui->textEdit->textCursor().insertFragment(fragment);
    //    ui->textEdit->setAlignment(Qt::AlignLeft);
    ui->textEdit->insertPlainText("Prompt: "+prompt +"\n");
    //调用qwen大模型回复 todo

    //llm->chat();
    //llm->warmup();
    std::string output_str;
    std::string prompt_str = prompt.toStdString();
    //    output_str = llm->response(prompt_str);
    //    llm->response(prompt_str, &response_buffer, "<eop>");
    //    QString output = QString::fromStdString(output_str);

    //    ui->textEdit->setAlignment(Qt::AlignRight);
    //    ui->textEdit->insertPlainText( "Qwen: "+ output +"\n");
    QString output = "sucess";
    qDebug() << "--------------->>> 准备调用模型！";
    //    output_str = llm->response(prompt_str, &response_buffer);
    output_str = llm->response(prompt_str);
    //    output_str = response_buffer.str();
    output = QString::fromStdString(output_str);

    //    if (llm.get() && llm->load_progress() >= 100) {


    //                //新的调用方式
    //                const char* input_str = prompt_str.data();
    //                auto chat = [&](std::string str) {
    //                    llm->response(str, &response_buffer);
    //                    output_str = response_buffer.str();
    //                    output = QString::fromStdString(output_str);
    //                };
    //                std::thread chat_thread(chat, input_str);
    //                chat_thread.detach();

    //    }

    //    ui->textEdit->setAlignment(Qt::AlignRight);
    ui->textEdit->insertPlainText( "Qwen: "+ output +"\n\n");
    //清空stringstream
    //    response_buffer.str("");
    llm->reset(); // 清空history，暂时不要多伦对话，真实可以固定多伦对话关联的轮次

}

void Widget::on_textEdit_2_selectionChanged()
{
    //    ui->textEdit_2->setInputMethodHints(Qt::ImhExclusiveInputMask);
    //    QInputMethod *keyboard = QGuiApplication::inputMethod();
    //    keyboard->show();

}
