import debugpy


def start_debug():
    debugpy.listen(5676)
    print("Wait for debugger!")
    debugpy.wait_for_client()
    print("Attached!")
