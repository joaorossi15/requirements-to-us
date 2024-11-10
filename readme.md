# Ethical Requirements to Ethical US

LLM-based tool powered by Mistral v0.3 that transforms high-level ethical requirements into actionable ethical user stories. Utilizing HuggingFace, retrieval-augmented generation (RAG), ChromaDB and LangChain, this tool bridges the gap between abstract ethical principles and practical implementation, ensuring ethical considerations are seamlessly integrated into software development processes.


## Features

- Ethical user stories generation
- Update on RAG database locally



## Run on Colab
- Open Collab
- Get the `main.ipynb` file from the repository
- Run the cells and change the `query` variable to the desired input

## Run Locally
**Recommended to run on google colab given the size of the model**

Clone the project

```bash
  git clone https://github.com/joaorossi15/requirements-to-us.git
```

Go to the project directory

```bash
  cd requirements-to-us
```

Create and start a virtual enviroment

```bash
  python3 -m venv env
  source bin/env/activate
```

Install dependencies

```bash
  pip3 install -r requirements.txt
```

Run the jupyter notebook

```bash
  jupyter-notebook main.ipynb
```


## Authors

- [@joaorossi15](https://www.github.com/joaorossi15)


## License

[MIT](https://choosealicense.com/licenses/mit/)

