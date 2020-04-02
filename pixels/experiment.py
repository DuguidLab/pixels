

import datetime
import json
import os


class Experiment:
    def __init__(self, mouse_ids, data_dir, meta_dir):
        if not isinstance(mouse_ids, (List, Tuple, Set)):
            mouse_ids = Tuple(mouse_ids)

        self.mouse_ids = mouse_ids
        self.sessions = []
        self.sessions_by_mouse_id = {}
        self.read_sessions(data_dir, meta_dir)

    def read_sessions(self, data_dir, meta_dir):
        data_dir = os.path.expanduser(data_dir)
        meta_dir = os.path.expanduser(meta_dir)
        available_sessions = os.listdir(data_dir)

        for mouse in mouse_ids:
            mouse_sessions = []
            for session in available_sessions:
                if mouse in session:
                    mouse_sessions.append(session)

            if mouse_sessions:
                with open(os.path.join(metadata, mouse + '.json'), 'r') as fd:
                    mouse_meta = json.load(fd)
                session_dates = [
                    datetime.datetime.strptime(s[0:6], '%y%m%d') for s in mouse_sessions
                ]
                for session in mouse_meta:
                    meta_date = datetime.datetime.strptime(session['date'], '%Y-%m-%d')
                    for index, ses_date in enumerate(session_dates):
                        if ses_date == meta_date:
                            if session.get('exclude', False):
                                continue
                            mouse_sessions[index] = pixels.Session(metadata=session)
            else:
                print(f'Found no sessions for: {mouse}')

            self.sessions.extend(mouse_sessions)
            self.sessions_by_mouse_id[mouse] = mouse_sessions

    def process_behaviour(self):
        for session in self.sessions:
            session.create_action_labels()
