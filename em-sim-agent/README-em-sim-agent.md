# Embedding Similarity Agent `em-sim-agent`

This agent will play the Wikipedia game using pre-trained language embedding models.
The plan is an agent that will compare the similarity of the pages using an embedding model and go to the page with the closest embeddings.

## TODO

What is needed to get this agent running?

**Data Prep**

* Link page title to page ID
* For a page ID return all the IDs of the pages reachable from it
* Get information from a certain page ID
    * Summary
    * Title
* Best solution score for any two pages
