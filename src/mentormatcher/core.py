import sys, os, re, json
from dotenv import load_dotenv
import langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI, AsyncOpenAI

import pandas as pd
import asyncio
import aiohttp

class MentorMatcher:
	"""
	MentorMatcher class for storing results
	Input: Model (open source, or API)
	Output: List/Table of matches
	"""
	def __init__(self, model, cv):
		self.model, self.cv = model, cv
	
	def _load_cv(self, cv_path):
		cv = PyPDF2.read(cv_path)

	
