from typing import Callable, Optional, Tuple, TypeAlias, Any
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.experimental.host_callback as hcb
from jaxtyping import PyTree
import os
import gzip
try:
	import _pickle as pickle #type:ignore
except:
	import pickle

State: TypeAlias = PyTree

dflt_transform_fn = lambda data: {"t": data["t"], "s": data["s_"]}
dflt_host_transform_fn = lambda data: data["s"]

class Simulator:

	#-------------------------------------------------------------------

	def __init__(
		self, 
		model: PyTree,
		save_pth: Optional[str]=None, 
		save_freq: int=50,
		transform_fn: Callable[[dict], dict]=dflt_transform_fn,
		host_transform_fn: Callable[[dict], Any]=dflt_host_transform_fn,
		zip_files: bool=False):

		"""
		Args:
			model: PyTree, callable and have initialize method implemented
			save_pth: directory in which data will be saved
			save_freq: frequency at which data is saved
			transforn_fn: function taking raw data dict and transforming it. Is called on xla side
			hist_transform_fn: function taking output from transform_fn and outputting 
		"""

		if save_pth is not None and not os.path.isdir(save_pth):
			os.makedirs(save_pth)
		elif save_pth is not None:
			print(f"path {save_pth} already exist, simulating can overwrite content")
		
		self.model = model
		self.save_pth = save_pth
		self.save_freq = save_freq
		self.transform_fn = transform_fn
		self.host_transform_fn = host_transform_fn
		self.zip_files = zip_files

	#-------------------------------------------------------------------

	def log(self, data: dict):
		"""
		save data to self.save_pth
		Args:
			data (dict): data 
		"""
		def tap_func(data, *_):
			if self.save_pth is None:
				return
			t = data["t"]
			filepath = f"{self.save_pth}/{t}"
			data = self.host_transform_fn(data)
			if not self.zip_files:
				with open(filepath+".pickle", "wb") as file:
					pickle.dump(data, file)
			else:
				with gzip.GzipFile(filepath+".zip", "wb") as file:
					pickle.dump(data, file)

		data = self.transform_fn(data)
		assert 't' in data.keys()
		hcb.id_tap(tap_func, data)
		return None

	#-------------------------------------------------------------------

	def simulate(self, steps: int, key: jax.Array=jr.key(1)):

		def _step(t: int, c: Tuple[State, jax.Array]):
			# --- sim step
			s, k = c
			k, _k = jr.split(k)
			s_ = self.model(s, _k)
			# --- log
			jax.lax.cond(
				t%self.save_freq == 0,
				lambda data: self.log(data),
				lambda data: None,
				{"s": s, "s_": s_, "t": t}
			)
			return s_, k

		key, key_init = jr.split(key)
		s0 = self.model.initialize(key_init)
		s = jax.lax.fori_loop(0, steps, _step, (s0, key))
		return s

	#-------------------------------------------------------------------

	def simulate_scan(self, steps: int, key: jax.Array=jr.key(1)):
		
		def _step(c, t):
			# --- sim step
			s, k = c
			k, _k = jr.split(k)
			s_ = self.model(s, _k)
			# --- log
			jax.lax.cond(
				t%self.save_freq == 0,
				lambda data: self.log(data),
				lambda data: None,
				{"s": s, "s_": s_, "t": t}
			)
			return (s_, k), s

		key, key_init = jr.split(key)
		s0 = self.model.initialize(key_init)
		s, S = jax.lax.scan(_step, (s0, key), jnp.arange(steps))
		return s, S

	#-------------------------------------------------------------------

	def load_files(self):
		""""""
		def _load(pth):
			if self.zip_files:
				with gzip.GzipFile(pth, "rb") as file:
					o = pickle.load(file)
				return o
			else:
				with open(pth, "rb") as file:
					o = pickle.load(file)
				return o
		files = os.listdir(self.save_pth)
		datas = [_load(f"{self.save_pth}/{file}") for file in files]
		return datas

	#-------------------------------------------------------------------



if __name__ == '__main__':
	from flowlenia.flowlenia_params import FlowLeniaParams, Config
	cfg = Config(X=32, Y=32)
	mdl = FlowLeniaParams(cfg, key=jr.key(2))
	sim = Simulator(mdl, "../flowlenia_saves_test", zip_files=True)
	sim.simulate(200)
	data = sim.load_files()