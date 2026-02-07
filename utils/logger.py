# -*- coding: utf-8 -*-
"""
日志工具模块
提供控制台和文件日志功能
"""
import logging
import sys
import os
from datetime import datetime


def setup_logger(
    name="hase",
    log_dir="logs",
    log_level=logging.INFO,
    console_output=True,
    file_output=True
):
    """
    配置并返回一个logger实例
    
    Args:
        name: logger名称
        log_dir: 日志文件保存目录
        log_level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
    
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if file_output:
        log_path = os.path.abspath(log_dir)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        # 日志文件名包含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_path, "{}_{}.log".format(name, timestamp))
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 按名称缓存的 logger 实例
_loggers = {}


def get_logger(name="hase"):
    """
    获取指定名称的 logger 实例（按名称缓存）
    
    Args:
        name: logger名称
    
    Returns:
        logger实例
    """
    if name not in _loggers:
        _loggers[name] = setup_logger(name=name)
    return _loggers[name]
