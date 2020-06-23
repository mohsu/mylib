import glob
import os
import shutil
from os.path import *

from loguru import logger

IMAGE_EXT = [".jpg", ".png"]


@logger.catch(reraise=True)
def chdir(path):
    logger.debug("Current path has changed to {}".format(os.path.abspath(path)))
    os.chdir(path)


@logger.catch(reraise=True)
def join(path, *paths):
    path = str(path)
    paths_s = []
    for i in range(len(paths)):
        paths_s.append(str(paths[i]))
    return os.path.join(path, *paths_s)


@logger.catch(reraise=True)
def check_pattern(pattern, filename, check_all=False):
    if isinstance(pattern, list):
        return check_patterns(pattern, filename, check_all)

    if (pattern is not None and pattern in filename) or pattern is None:
        pattern_check = True
    else:
        pattern_check = False

    return pattern_check


@logger.catch(reraise=True)
def check_patterns(patterns, filename, check_all=False):
    for pattern in patterns:
        is_contain = check_pattern(pattern, filename, False)
        if check_all and not is_contain:
            return False
        elif is_contain:
            return True

    if check_all:
        return True
    else:
        return False


@logger.catch(reraise=True)
def check_extension(extension, filename):
    if extension is None:
        return True

    if isinstance(extension, list):
        return check_extensions(extension, filename)

    if extension == os.path.splitext(filename)[-1]:
        extension_check = True
    else:
        extension_check = False

    return extension_check


@logger.catch(reraise=True)
def check_extensions(extensions, filename):
    for extension in extensions:
        if check_extension(extension, filename):
            return True
    return False


@logger.catch(reraise=True)
def list_dir(_dir, sort=False, recursive=False, pattern=None,
             extension=None, full_path=True, abs_path=False, exclude_pattern="\x00", check_all_pattern=False):
    paths = []
    if not os.path.exists(_dir):
        return paths

    if abs_path:
        _dir = os.path.abspath(_dir)

    if not recursive:
        files = os.listdir(_dir)
        if sort:
            files = sorted(files)

        for i in range(len(files)):
            if check_pattern(pattern, files[i], check_all_pattern) and \
                    check_extension(extension, files[i]) and \
                    (exclude_pattern and not check_pattern(exclude_pattern, files[i], check_all_pattern)):
                if full_path:
                    paths.append(os.path.join(_dir, files[i]))
                else:
                    paths.append(files[i])
    else:
        for root, directories, filenames in os.walk(_dir):
            if extension == "":
                for directory in directories:
                    if check_pattern(pattern, directory, check_all_pattern) and \
                            not check_pattern(exclude_pattern, directory, check_all_pattern):
                        if full_path:
                            paths.append(os.path.join(root, directory))
            else:
                for filename in filenames:
                    if check_pattern(pattern, filename, check_all_pattern) and \
                            check_extension(extension, filename) and \
                            not check_pattern(exclude_pattern, filename, check_all_pattern):
                        if full_path:
                            paths.append(os.path.join(root, filename))
                        else:
                            paths.append(filename)

    return paths


@logger.catch(reraise=True)
def make_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)
        logger.debug(f"Directory {os.path.abspath(_dir)} is created")
    return _dir


@logger.catch(reraise=True)
def dirname(path, depth=1, full_path=True):
    _dir = os.path.dirname(path)
    if depth > 1:
        for i in range(depth - 1):
            _dir = os.path.dirname(_dir)
    if not full_path:
        _dir = os.path.basename(_dir)
    return _dir


@logger.catch(reraise=True)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'mytest', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


@logger.catch(reraise=True)
def rm(full_path):
    if not exists(full_path):
        return
    if os.path.isfile(full_path):
        os.unlink(full_path)
    else:
        shutil.rmtree(full_path)


@logger.catch(reraise=True)
def cleanup_folder_by_time(folder_s, num_keep):
    contents = list_dir(folder_s, exclude_pattern=".gitkeep", full_path=True)
    files = []
    for content in contents:
        if os.path.isfile(content):
            files.append(content)
        else:
            cleanup_folder_by_time(content, num_keep)

    files = sorted(files, key=os.path.getctime, reverse=True)  # newest at front (0)
    for i in range(num_keep, len(files)):
        rm(files[i])
        logger.debug(f"Removing {files[i]}")

    # if directory is empty, then delete
    if not list_dir(folder_s, recursive=True):
        rm(folder_s)
        logger.debug(f"Removing {folder_s}")


@logger.catch(reraise=True)
def get_filename(path):
    filename = os.path.basename(path)
    filename, _ = os.path.splitext(filename)
    return filename


@logger.catch(reraise=True)
def make_sub_dirs(_dir, sub_dirs):
    for sub_dir in sub_dirs:
        make_dir(join(_dir, sub_dir))
