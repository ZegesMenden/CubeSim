
def castCheck(val: any, cast: type, name: str) -> any:
    try: return cast(val)
    except (ValueError, TypeError): raise TypeError(f"{name} must be a {cast}-like object")
    except Exception as e: raise e
