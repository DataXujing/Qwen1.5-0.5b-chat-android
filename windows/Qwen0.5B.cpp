// Qwen0.5B.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "llm.hpp"
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main() {
	
	std::string model_dir ="./qwen_model";

	std::cout << "model path is " << model_dir << std::endl;
	std::unique_ptr<Llm> llm(Llm::createLLM(model_dir,"qwen1.5-0.5b-chat"));

	llm->load(model_dir);
	//llm->chat();
	//llm->warmup();
	std::string prompt = "写一首关于清明的诗";

	string output;
	std::cout << prompt << std::endl;
	output = llm->response(prompt);
	std::cout << "--------------------" << std::endl;

	std::string prompt1 = "马云是谁";
	std::cout << prompt1 << std::endl;
	output = llm->response(prompt1);

	return 0;
}
