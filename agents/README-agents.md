# Wiki Game Agent `agents`

The agents in this package will play the Wikipedia game. There is an abstract class of `Agent`, which the derived classes will inherit from.

## Data
This package is currently only designed to be implemented with Wikipedia data, which is not saved to the repo due to size. A class called `AgentTools` has been designed to load and manage the data. The data contain unique IDs for each page in the Wikipedia data dump and through an instance of `AgentTools` it is possible to get the title, summary and links on any page given its unique ID.

## Agent Classes

Each dervied class will be initialised by passing an instance of `AgentTools` and also the maximum number of rounds that can be played in each game. A game is when an agent is given one pair of start and end (or source and target) pages and it has to navigate from the start to the end. If it fails to do so within the maximum number of steps (or rounds), then the game is logged as a failure.

### GreedyEmbeddingAgent
This implementation uses a pre-trained sentence transformer model to play the Wikipedia game. It uses the model to encode the summaries of all links on a page, and selects the one that is most similar to the encoded summary of the target page. This agent is not allowed to navigate back to a page it has already visited, as this approach will cause to enter a loop; therefore, this agent may also fail to complete a game if it reaches a page where there are no links available that it has not already visited.

### LLMPromptAgent
Not yet implemented, but this will use Azure's OpenAI service to play the game with a GPT engine.
