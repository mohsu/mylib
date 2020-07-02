# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: telegram_bot.py 
@time: 2019/10/18
"""

# python packages
import requests

# 3rd-party packages
# from configobj import ConfigObj
# from loguru import logger

# self-defined packages
# from utils import os_path

# global variables
# log_path = "./log/"
# config_path = "./config/"

# set up log & config
# logger.add(log_path, rotation="5 MB", retention="1 week") # filter=lambda record: record["extra"].get("name") == ""
# logger = logger.bind(name="")
# config = ConfigObj(config_path)


class TgBot:
    def __init__(self, token):
        self.token = token
        self.send_photo_url = "https://api.telegram.org/bot{}/sendPhoto".format(token)

    def send_message(self, telegram_group_chat_id, text):
        """
        Send message from telegram bot in group
        :param telegram_group_chat_id:
        :param text:
        :return:
        """
        r = requests.get("https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(self.token,
                                                                                                telegram_group_chat_id,
                                                                                                text))
        return r

    def send_photo(self, image_io, telegram_group_chat_id, caption=None):
        """
        Send photo from telegram bot in group
        :param image_io: files = {'photo': open(image_path), 'rb')}
        :param telegram_group_chat_id
        :param caption
        :return: response
        """
        data = {'chat_id': telegram_group_chat_id}
        if caption:
            data["caption"] = caption

        r = requests.post(self.send_photo_url, files={'photo': image_io}, data=data)
        image_io.close()
        return r
