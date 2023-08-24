import debugpy


def start_debug():
    debugpy.listen(5679)
    print("Wait for debugger!")
    debugpy.wait_for_client()
    print("Attached!")
