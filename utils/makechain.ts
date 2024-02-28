import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ChatPromptTemplate } from 'langchain/prompts';
import { RunnableSequence } from 'langchain/schema/runnable';
import { StringOutputParser } from 'langchain/schema/output_parser';
import type { Document } from 'langchain/document';
import type { VectorStoreRetriever } from 'langchain/vectorstores/base';

const CONDENSE_TEMPLATE = `Với cuộc trò chuyện sau và một câu hỏi tiếp theo, hãy chuyển câu hỏi tiếp theo thành một câu hỏi độc lập.

<chat_history>
  {chat_history}
</chat_history>

Câu hỏi tiếp theo: {question}
Câu hỏi độc lập:`;

const QA_TEMPLATE = `Bạn là một nhà nghiên cứu chuyên gia. Sử dụng các đoạn văn bản sau để trả lời câu hỏi ở cuối.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết. ĐỪNG cố gắng tạo ra một câu trả lời.
Nếu câu hỏi không liên quan đến ngữ cảnh hoặc lịch sử trò chuyện, hãy lịch sự trả lời rằng bạn chỉ trả lời câu hỏi liên quan đến ngữ cảnh.

<context>
  {context}
</context>

<chat_history>
  {chat_history}
</chat_history>

Câu hỏi: {question}
Câu trả lời hữu ích dưới dạng markdown:`;

const combineDocumentsFn = (docs: Document[], separator = '\n\n') => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join(separator);
};

export const makeChain = (retriever: VectorStoreRetriever) => {
  const condenseQuestionPrompt =
    ChatPromptTemplate.fromTemplate(CONDENSE_TEMPLATE);
  const answerPrompt = ChatPromptTemplate.fromTemplate(QA_TEMPLATE);

  const model = new ChatOpenAI({
    temperature: 0, 
    modelName: 'gpt-3.5-turbo',
  });

  const standaloneQuestionChain = RunnableSequence.from([
    condenseQuestionPrompt,
    model,
    new StringOutputParser(),
  ]);

  const retrievalChain = retriever.pipe(combineDocumentsFn);

  const answerChain = RunnableSequence.from([
    {
      context: RunnableSequence.from([
        (input) => input.question,
        retrievalChain,
      ]),
      chat_history: (input) => input.chat_history,
      question: (input) => input.question,
    },
    answerPrompt,
    model,
    new StringOutputParser(),
  ]);

  const conversationalRetrievalQAChain = RunnableSequence.from([
    {
      question: standaloneQuestionChain,
      chat_history: (input) => input.chat_history,
    },
    answerChain,
  ]);

  return conversationalRetrievalQAChain;
};
