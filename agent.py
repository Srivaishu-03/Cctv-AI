from visual import search_text

def ai_agent(query):

    query = query.lower()

    if "car" in query:
        return search_text("car")

    elif "person" in query:
        return search_text("person")

    elif "bus" in query:
        return search_text("bus")

    elif "truck" in query:
        return search_text("truck")
    
    elif "helmet" in query:
        return search_text("helmet")
    else:
        return search_text(query)