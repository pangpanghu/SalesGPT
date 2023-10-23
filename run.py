import argparse  # 导入处理命令行参数的库
import json  # 导入json处理库

from dotenv import load_dotenv  # 从dotenv导入函数加载环境变量
from langchain.chat_models import ChatLiteLLM  # 从langchain.chat_models导入ChatLiteLLM模型

from salesgpt.agents import SalesGPT  # 从salesgpt.agents导入SalesGPT模型

load_dotenv()  # 加载.env文件中的环境变量

if __name__ == "__main__":  # 当前脚本作为主程序执行
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser(description="Description of your program")

    # 添加命令行参数，这个就是运行python3 run.py --config config.json --verbose True --max_num_turns 10 时的参数
    parser.add_argument(
        "--config", type=str, help="Path to agent config file", default="./examples/example_cn_agent_setup.json"
    )
    parser.add_argument("--verbose", type=bool, help="Verbosity",
                        default=False)
    parser.add_argument(
        "--max_num_turns",
        type=int,
        help="Maximum number of turns in the sales conversation",
        default=20,
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取命令行参数的值
    
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns
    
    '''
    config_path = "examples/sample_product_catalog_1.txt"
    verbose = True
    max_num_turns = 20
    '''
    
    # 初始化ChatLiteLLM模型
    llm = ChatLiteLLM(temperature=0, model_name="gpt-3.5-turbo-instruct")

    #测试用 config_path == ""
    
    # 检查是否提供了代理配置文件
    if config_path == "":
        print("No agent config specified, using a standard config")
        # 为了与JSON配置保持一致，将布尔值用字符串表示
        USE_TOOLS = "True"
        
        if USE_TOOLS == "True":
            sales_agent = SalesGPT.from_llm(
                llm,
                use_tools=USE_TOOLS,
                product_catalog="examples/sample_product_catalog_1.txt",
                salesperson_name="钢铁侠",
                verbose=verbose,
            )
        else:
            sales_agent = SalesGPT.from_llm(llm, verbose=verbose)
    else:
        with open(config_path, "r", encoding="UTF-8") as f:
            config = json.load(f)
        
        #print(f"Agent config {config}")
        
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)

    # 初始化代理
    sales_agent.seed_agent()
    print("=" * 10) # 打印10个等号作为对话间的分隔符
    cnt = 0
    
    # 与销售代理进行对话，直到达到最大回合数
    while cnt != max_num_turns:
        cnt += 1
        if cnt == max_num_turns:
            print("Maximum number of turns reached - ending the conversation.")
            break
        
        sales_agent.step()

        # 如果会话历史中存在“<END_OF_CALL>”则结束对话
        if "<END_OF_CALL>" in sales_agent.conversation_history[-1]:
            print("Sales Agent determined it is time to end the conversation.")
            break
        # 获取人类的输入
        human_input = input("Your response: ")
        sales_agent.human_step(human_input)
        print("=" * 10)
