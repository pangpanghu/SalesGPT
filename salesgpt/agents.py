from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from langchain.chat_models import ChatLiteLLM
from langchain.llms.base import create_base_retry_decorator
from litellm import acompletion
from pydantic import Field

# 导入salesgpt库相关类

from salesgpt.chains import SalesConversationChain, StageAnalyzerChain
from salesgpt.logger import time_logger
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.prompts import SALES_AGENT_TOOLS_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.tools import get_tools, setup_knowledge_base

# 定义重试装饰器函数
def _create_retry_decorator(llm: Any) -> Callable[[Any], Any]:
    import openai
 # 定义可能的错误类型
    errors = [
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries)

# 定义SalesGPT类
class SalesGPT(Chain):
    """Controller model for the Sales Agent."""
    
    # 定义一些默认属性和它们的默认值
    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    knowledge_base: Union[RetrievalQA, None] = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    model_name: str = "gpt-3.5-turbo-0613"

    use_tools: bool = False
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "销售"
    company_name: str = "好好玩游戏公司"
    company_business: str = "提供做好玩的游戏"
    company_values: str = "好玩的游戏都在好好玩游戏公司"
    conversation_purpose: str = "find out whether they are looking to achieve happiness via buying a game we have."
    conversation_type: str = "message chat"
    
    
    # 根据给定的key返回对应的会话阶段
    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    # 返回空列表，因为此类没有定义input_keys属性
    @property
    def input_keys(self) -> List[str]:
        return []
    # 返回空列表，因为此类没有定义output_keys属性
    @property
    def output_keys(self) -> List[str]:
        return []

    # 初始化对话
    @time_logger
    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []
        print(f"seed_agent: {self.current_conversation_stage}")

    # 确定会话阶段
    @time_logger
    def determine_conversation_stage(self): # 定义一个名为“determine_conversation_stage”的方法，该方法用于确定对话阶段
        
        self.conversation_stage_id = self.stage_analyzer_chain.run(
            # 使用stage_analyzer_chain对象的run方法来确定对话阶段，并将结果赋值给self.conversation_stage_id
            conversation_history="\n".join(self.conversation_history).rstrip("\n"),
            # 将self.conversation_history列表的元素连接成一个字符串，每个元素之间用换行符分隔，然后删除最后一个换行符
            conversation_stage_id=self.conversation_stage_id,
            # 使用当前的self.conversation_stage_id作为参数
            conversation_stages="\n".join(
                # 将CONVERSATION_STAGES字典中的键和值连接成一个字符串，并使用换行符分隔
                [
                    str(key) + ": " + str(value)
                    for key, value in CONVERSATION_STAGES.items()
                ]
            ),
        )

        print(f"Conversation Stage ID: {self.conversation_stage_id}")
        # 打印当前的对话阶段ID

        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )
        # 调用retrieve_conversation_stage方法来获取当前对话阶段的具体描述，并赋值给self.current_conversation_stage

        print(f"Conversation Stage: {self.current_conversation_stage}")
        # 打印当前的对话阶段描述
  
    # 处理人类的输入内容到conversation_history
    def human_step(self, human_input):

        human_input = "User: " + human_input + " <END_OF_TURN>"
        # 将输入字符串前后添加额外的文本，前面添加“User: ”，后面添加“ <END_OF_TURN>”

        self.conversation_history.append(human_input)
        
        print("human_step")
        print(f"Conversation Stage: {self.current_conversation_stage}")
        
        # 将处理后的human_input添加到self.conversation_history列表的末尾
    
    # 定义一个名为“step”的方法，该方法接收一个默认值为False的名为“stream”的布尔参数 可能是一个同步方法。
    
    @time_logger
    def step(self, stream: bool = False):
        
        """
        Args:
            stream (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        
        print("Step")
        print(f"Conversation Stage: {self.current_conversation_stage}")
        
        if not stream:
            
            print("Not Stream\n")
            
            self._call(inputs={})
        else:
            
            print("Stream\n")
            
            return self._streaming_generator()
    
    # 方法可能是一个异步版本的 step，用于处理流或非流数据
    @time_logger
    def astep(self, stream: bool = False):
        """
        Args:
            stream (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not stream:
            self._acall(inputs={})
        else:
            return self._astreaming_generator()

    @time_logger
    def acall(self, *args, **kwargs):
        raise NotImplementedError("This method has not been implemented yet.")

    @time_logger
    def _prep_messages(self):
        """
        Helper function to prepare messages to be passed to a streaming generator.
        """
        prompt = self.sales_conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    salesperson_name=self.salesperson_name,
                    salesperson_role=self.salesperson_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()

        message_dict = {"role": "system", "content": inception_messages[0].content}

        if self.sales_conversation_utterance_chain.verbose:
            print("\033[92m" + inception_messages[0].content + "\033[0m")
        return [message_dict]

    @time_logger
    def _streaming_generator(self):
        """
        Sometimes, the sales agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output. 例如，如果我们想对部分LLM输出进行文本转语音

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._streaming_generator()
        # Now I can loop through the output in chunks:
        >> for chunk in streaming_generator:
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return self.sales_conversation_utterance_chain.llm.completion_with_retry(
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=self.model_name,
        )

    async def acompletion_with_retry(self, llm: Any, **kwargs: Any) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(llm)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            # Use OpenAI's async api https://github.com/openai/openai-python#async-api
            return await acompletion(**kwargs)

        return await _completion_with_retry(**kwargs)

    async def _astreaming_generator(self):
        """
        Asynchronous generator to reduce I/O blocking when dealing with multiple
        clients simultaneously.

        Sometimes, the sales agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._astreaming_generator()
        # Now I can loop through the output in chunks:
        >> async for chunk in streaming_generator:
            await chunk ...
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return await self.acompletion_with_retry(
            llm=self.sales_conversation_utterance_chain.llm,
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=self.model_name,
        )


    # 定义一个名为“_call”的方法，该方法接收一个名为“inputs”的字典参数。它模拟了一个销售代理进行沟通的过程。
    def _call(self, inputs: Dict[str, Any]) -> None: 
        """Run one step of the sales agent."""  # 文档字符串：运行销售代理的一步操作

        # Generate agent's utterance
        # if use tools
        if self.use_tools:  # 如果使用工具
            ai_message = self.sales_agent_executor.run(  # 从 "sales_agent_executor" 获取 AI 的消息
                input="",  # 输入为空
                conversation_stage=self.current_conversation_stage,  # 当前的沟通阶段
                conversation_history="\n".join(self.conversation_history),  # 之前的沟通历史
                salesperson_name=self.salesperson_name,  # 销售代表的名字
                salesperson_role=self.salesperson_role,  # 销售代表的角色
                company_name=self.company_name,  # 公司名称
                company_business=self.company_business,  # 公司业务
                company_values=self.company_values,  # 公司价值观
                conversation_purpose=self.conversation_purpose,  # 沟通的目的
                conversation_type=self.conversation_type,  # 沟通的类型
            )

        else:  # 否则
            ai_message = self.sales_conversation_utterance_chain.run(  # 从 "sales_conversation_utterance_chain" 获取 AI 的消息
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
            )

        # Add agent's response to conversation history
        agent_name = self.salesperson_name  # 代理的名字
        ai_message = agent_name + ": " + ai_message  # 将代理的名字加到 AI 消息前面
        if "<END_OF_TURN>" not in ai_message:  # 如果消息中没有 "<END_OF_TURN>"
            ai_message += " <END_OF_TURN>"  # 在消息末尾添加 "<END_OF_TURN>"
        self.conversation_history.append(ai_message)  # 将 AI 消息添加到沟通历史中
        print(ai_message.replace("<END_OF_TURN>", ""))  # 打印消息，同时移除 "<END_OF_TURN>"
        return {}  # 返回一个空字典
    
    
    # 使用LLM初始化SalesGPT控制器
    @classmethod
    @time_logger  # 使用时间记录器装饰器
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""
        
        # 从LLM中创建一个StageAnalyzerChain实例
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        # 检查是否有自定义的提示
        if (
            "use_custom_prompt" in kwargs.keys()
            and kwargs["use_custom_prompt"] == "True"
        ):
            # 如果有，则深度复制这个自定义提示
            use_custom_prompt = deepcopy(kwargs["use_custom_prompt"])
            custom_prompt = deepcopy(kwargs["custom_prompt"])

            # 清除这两个关键字参数，避免后续冲突
            del kwargs["use_custom_prompt"]
            del kwargs["custom_prompt"]

            # 使用自定义提示创建一个SalesConversationChain实例
            sales_conversation_utterance_chain = SalesConversationChain.from_llm(
                llm,
                verbose=verbose,
                use_custom_prompt=use_custom_prompt,
                custom_prompt=custom_prompt,
            )
        else:
            # 不使用自定义提示创建一个SalesConversationChain实例
            sales_conversation_utterance_chain = SalesConversationChain.from_llm(
                llm, verbose=verbose
            )

        # 检查是否使用工具
        if "use_tools" in kwargs.keys() and (
            kwargs["use_tools"] == "True" or kwargs["use_tools"] == True
        ):
            # 设置代理使用的工具
            product_catalog = kwargs["product_catalog"]  # 产品目录
            knowledge_base = setup_knowledge_base(product_catalog)  # 设置知识库
            tools = get_tools(knowledge_base)  # 获取工具

            # 创建一个针对工具的自定义提示模板
            prompt = CustomPromptTemplateForTools(
                template=SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # 动态生成`agent_scratchpad`，`tools`和`tool_names`变量
                # 需要`intermediate_steps`变量
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            # 警告：这个输出解析器还不可靠
            ## 它基于LLM的输出做了假设，可能会抛出错误
            output_parser = SalesConvoOutputParser(ai_prefix=kwargs["salesperson_name"])

            # 创建一个使用工具的代理
            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
            )

            # 创建一个执行器，结合代理和工具
            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools, tools=tools, verbose=verbose
            )
        else:
            sales_agent_executor = None
            knowledge_base = None

        # 返回一个新的SalesGPT实例
        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            knowledge_base=knowledge_base,
            model_name=llm.model,
            verbose=verbose,
            **kwargs,
        )
