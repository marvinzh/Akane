import strings


def model_reporter(slient_mode, title, profile, detail):
    dic = strings.MODEL_REPORTER_EN

    profile_list = list(profile.keys())
    detail_list = list(detail.keys())

    if not slient_mode:
        print(title + ":")
        print("-------------------------------------------------------------")
        for i in range(len(profile_list)):
            print("%20s:\t\t%s" % (profile_list[i], profile.get(profile_list[i])))

        print("-------------------------------------------------------------")
        for i in range(len(detail_list)):
            print("%20s:\t\t%s" % (detail_list[i], detail.get(detail_list[i])))



class Reporter(object):
    """docstring for Messager"""

    def __init__(self, silent_mode):
        super(Reporter, self).__init__()
        self.silent_mode = silent_mode

    def report(self, str=""):
        if not self.silent_mode:
            print(str)



