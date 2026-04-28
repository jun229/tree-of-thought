def get_task(name: str):
    if name == "game24":
        from tot.tasks.game24 import Game24Task
        return Game24Task()
    raise ValueError(f"task {name!r} not recognized")
