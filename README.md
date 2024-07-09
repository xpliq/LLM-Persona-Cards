# LLM-Persona-Cards

Creating Persona Cards by leveraging LLMs to analyze text from responses.



## Design Goal
This system aims to take a source of input in the form of responses, such as notes, documents, or transcripts, and using a prompted LLM to extract relevant labels from the input as data to use later for development of Persona Cards.
![](https://ibb.co/Vm8kPJk)

## Why Persona Cards?
- Persona cards can be used to feed back into an LLM to take the role of the persona.
- Personafied representation of input data.


# Usage
- The design is currently displayed using OpenAI. 
- Usage of other LLMs require modifications.
- This repository serves as a demonstration of the system, by assuming questions and asking them towards LLMs with a prompted persona.

## Installation
These next steps are assuming that you have python installed, and you have all files downloaded within the desired directory.

### Install Requirements
1. Create a virtual environment in your project directory.
```
python -m venv .venv
```

2. Activation of environment on Linux and MacOS:
```
source .env/bin/activate
```
3. Activation of environment on Windows:
```
.env/Scripts/activate
```
4. Install required packages and libraries from the 'requirements.txt' file
```
!pip  install  -r  "requirements.txt"
```

## How to Run
Here is example code you can write to simply run the system. Your inputs will be recorded through your respective command line. You can refer to '```example.py```' for the exact example used below.
```
from PersonaCard import  LLMProcessor
import os
from dotenv import  load_dotenv

load_dotenv()
api_key  = os.environ.get("OPENAI_API_KEY")

# Initialize the processor with your OpenAI API key
card  =  LLMProcessor(api_key)
card.run()
```


# Example

## Design Concerns
- There is randomness in responses. The demonstration does not use an LLM agent, therefore a new synthetic response is generated each time.



