# NextTail AI - AI Front End Developer Suite

NextTail AI is a comprehensive AI Front End Developer Suite designed to address the challenges associated with converting CSS to Tailwind CSS. Our solution incorporates cutting-edge technologies to automate the conversion process, enhance user interface (UI) development, and provide extensive documentation support.

## Features

### Automated Conversion Tool
NextTail AI includes a powerful tool that automatically translates CSS code into Tailwind CSS. This streamlines the transition process for developers, making it more efficient and error-free. The tool not only converts CSS code but also translates UI designs directly into code.

### UI to Code
NextTail AI goes beyond traditional conversion by offering a feature that converts images into code. This feature accelerates the development process by generating code directly from UI mockups or designs.

### Documentation Chat
We understand the importance of clear documentation. NextTail AI provides a unique documentation chat, currently focused on Next JS and Tailwind. In the future, we plan to expand chat support to include other popular tech stacks, providing users with instant assistance and guidance.

## Technology Stack

NextTail AI is built on the **Django framework** and is hosted on **Digital Ocean** for reliable and scalable performance. The project utilizes **Astra DB** for storing vector embeddings, employing **Cohere's embed-english-v3.0 model** for creating embeddings. The retrieval process benefits from **Cohere's reranker**, ensuring fast and accurate document extraction based on queries from the database.

To enhance overall functionality, **Langchain** and **Llama Index** plays an integral part. The main Language Model (LLM) driving the CSS to Tailwind conversion is **Gemini Pro** from Google. Langchain is used to streamline its performance, and an HTML preview feature is included. **Gemini Pro Vision** along with Llama Index is used to bring _UI to Code_ Feature to life.

Evaluation metrics are crucial for performance analysis, and NextTail AI utilizes  **TruLens** for evaluation of the Mulitmodal. This combination, chained using Llama Index, ensures thorough evaluation of the multimodal capabilities.

For storage needs, NextTail AI leverages Amazon S3 bucket, ensuring secure and efficient data storage.



## Future Developments

In our continuous commitment to improvement, we have ambitious plans for NextTail AI's future:

- **Optimization for Daily Use:** We aim to enhance the speed and overall efficiency of NextTail AI, making it a reliable tool for daily development tasks.

- **UI Overhaul:** Expect significant changes to the user interface, aimed at improving user experience (UX) and making the suite more user-friendly.

We are dedicated to staying at the forefront of AI-driven development tools, providing developers with a seamless and efficient experience in building outstanding user interfaces. Your feedback is invaluable as we strive to make NextTail AI an essential part of your front-end development toolkit.
