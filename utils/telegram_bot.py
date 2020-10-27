# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: telegram_bot.py 
@time: 2019/10/18
"""

# python packages
import requests
from typing import Any, Union, Optional


# 3rd-party packages


class TgBot:
    def __init__(self, token: Any):
        self.token = token
        self.send_photo_url = "https://api.telegram.org/bot{}/sendPhoto".format(token)

    def send_message(self, telegram_group_chat_id: Union[str, int],
                     text: Any, parse_mode: Optional[Any] = None):
        """
        Send message from telegram bot in group
        """
        request = "https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(self.token,
                                                                                         telegram_group_chat_id,
                                                                                         text)
        if parse_mode is not None:
            request += "&parse_mode={}".format(parse_mode)
        r = requests.get(request)

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
