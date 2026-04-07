from typing import TypedDict, Annotated

from langgraph.graph import add_messages


def foo1():
    print("foo1")
    pass


class SearchState(foo1):
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: str


def main():
    print("main")
    state = SearchState()
    print(type(state))


if __name__ == '__main__':
    main()
