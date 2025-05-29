
def save_to_db(session, data, log_msg: str):
    session.add(data)
    session.commit()
    print(log_msg)
