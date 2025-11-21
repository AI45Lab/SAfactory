# DiscoveryFeed.py
# A news feed for status updates from agents, as well as scientific articles published by agents (or that are preexisting in a given scenario)

import json
import copy


class DiscoveryFeed:
    # Constructor
    def __init__(self, filenameIn: str = None):
        # There are two types of posts: scientific discovery articles, and agent status update posts.
        self.articles = []
        self.updatePosts = []

        self.uniquePostIDs = 1

        # Load the file (if supplied)
        if (filenameIn != None):
            self.loadFromFile(filenameIn)

    # Get a unique post ID
    def getUniquePostID(self):
        self.uniquePostIDs += 1
        return self.uniquePostIDs - 1

    #
    # Load articles and update posts from a JSON file
    #
    def loadFromFile(self, filenameIn: str):
        # Load the file
        data = {}
        try:
            with open(filenameIn, 'r') as file:
                data = json.load(file)
        except:
            print("ERROR: DiscoveryFeed: Could not load file containing articles and updates: " + filenameIn)
            exit(1)

        # Add the articles
        for article in data["articles"]:
            article["postID"] = self.getUniquePostID()      # Replace the unique post ID with a new one
            self.articles.append(article)

        # Sort the articles by step (ascending)
        self.articles.sort(key=lambda x: x["step"], reverse=False)

        # Add the update posts
        for post in data["updatePosts"]:
            post["postID"] = self.getUniquePostID()         # Replace the unique post ID with a new one
            self.updatePosts.append(post)

        # Sort the update posts by step (ascending)
        self.updatePosts.sort(key=lambda x: x["step"], reverse=False)


    #
    #   Add posts
    #
    ## TODO: (PJ) I think `signals` should default to [], not None (just needs to be tested to make sure it doesn't break anything)
    def addUpdatePost(self, curStep:int, authorName:str, content:str, signals:list = None):
        postID = self.getUniquePostID()
        self.updatePosts.append({"step": curStep, "author": authorName, "content": content, "signals": signals, "postID": postID, "type": "update"})
        return postID

    def addArticle(self, curStep:int, authorName:str, title:str, content:str):
        postID = self.getUniquePostID()
        self.articles.append({"step": curStep, "author": authorName, "title": title, "content": content, "signals": [], "postID": postID, "type": "article"})
        return postID

    #
    #   Get posts
    #
    def getPosts(self):
        return self.updatePosts

    def getArticles(self):
        return self.articles


    def getPostByID(self, postID:int):
        for article in self.articles:
            if (article["postID"] == postID):
                return article

        for post in self.updatePosts:
            if (post["postID"] == postID):
                return post

        return None

    #
    #   Get recent posts
    #
    def getRecentPosts(self, curStep:int):
        posts = []
        for post in self.updatePosts:
            if (post["step"] == curStep):
                posts.append(post)
        return posts

    #
    #   Collect all signals from posts made in the current step
    #


    # Collect all the signals from posts and articles made in the current step
    def getSignalsFromPosts(self, curStep:int):
        signals = set()
        for post in self.updatePosts:
            if (post["step"] == curStep):
                for signal in post["signals"]:
                    signals.add(signal)

        for article in self.articles:
            if (article["step"] == curStep):
                for signal in article["signals"]:
                    signals.add(signal)

        return list(signals)


    def getRecentPosts(self, curStep:int, lastNSteps:int = 3):
        # Get all posts posted within lastNSteps of curStep
        posts = []
        articles = []

        # Get the posts
        for post in self.updatePosts:
            if ((post["step"] >= (curStep - lastNSteps)) and (post["step"] <= curStep)):
                # Add the post, but with the signals removed
                postCopy = copy.deepcopy(post)
                del postCopy["signals"]
                posts.append(postCopy)

        # Get the articles (but just post the step, id, titles and authors)
        for article in self.articles:
            if ((article["step"] >= (curStep - lastNSteps)) and (article["step"] <= curStep)):
                articles.append({"step": article["step"], "postID": article["postID"], "title": article["title"], "author": article["author"]})

        # Packed
        packed = {
            "description": "This section contains recent posts (from the last few steps) on the Discovery Feed social media platform.",
            "posts": posts,
            "scientific_articles": articles
        }

        return packed

    #
    #   Export to dictionary
    #
    def toDict(self):
        return {"articles": self.articles, "updatePosts": self.updatePosts}
