import numpy as np, os, sys, re, glob, subprocess, math, unittest, shutil, time, string, logging, gc
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
np.set_printoptions(precision=4)
from time import gmtime, strftime
from random import shuffle, choice, sample, choices
from itertools import product
from functools import partial
import inspect
from LLM_AS2 import init_algorithms_pool_V2,function_crossover,function_mutation,Post_processing,generate_novel_cluster
import pickle 
import textwrap
import shutil
import traceback




base_folder = './'
try:
	os.mkdir(base_folder+'log')
	os.mkdir(base_folder+'agent_pool')
	os.mkdir(base_folder+'job_pool')
	os.mkdir(base_folder+'result_pool')
except:
	pass


current_time = strftime("%Y%m%d_%H%M%S", gmtime())

log_name = 'sim_{}.log'.format(sys.argv[1])

# Specify the source and destination file paths
source_log_path = './'+log_name



logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG,
										format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:  %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)






class DummyIndv(object): pass


evoluation_goal = 'data.npy'

class Individual(object):
	def __init__(self, adj_matrix=None, scope=None, **kwargs):
		super(Individual, self).__init__()
		if adj_matrix is None:
			self.adj_matrix = kwargs['adj_func'](**kwargs)
		else:
			self.adj_matrix = adj_matrix
		self.scope = scope
		self.sparsityB = np.sum(self.adj_matrix[np.triu_indices(self.adj_matrix.shape[0], 1)]>0)
		self.parents = kwargs['parents'] if 'parents' in kwargs.keys() else None
		self.repeat = kwargs['evaluate_repeat'] if 'evaluate_repeat' in kwargs.keys() else 1
		self.iters = kwargs['max_iterations'] if 'max_iterations' in kwargs.keys() else 10000
		self.dim = self.adj_matrix.shape[0]
		self.adj_matrix[np.tril_indices(self.dim, -1)] = self.adj_matrix.transpose()[np.tril_indices(self.dim, -1)]
		adj_matrix_k = np.copy(self.adj_matrix)
		adj_matrix_k[adj_matrix_k==0] = 1

		self.present_elements = np.prod(np.diag(adj_matrix_k))
		self.actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(self.dim) ])
		self.sparsity = self.actual_elements/self.present_elements

		self.rse_therhold=kwargs['rse_therhold'][0]
		self.Adam_Step=kwargs['Adam_Step'][0]



	def deploy(self, sge_job_id):
		try:
			path = base_folder+'/job_pool/{}.npz'.format(sge_job_id)
			np.savez(path, adj_matrix=self.adj_matrix, scope=self.scope, repeat=self.repeat, iters=self.iters,rse_TH=self.rse_therhold,Adam_Step=self.Adam_Step)
			self.sge_job_id = sge_job_id
			return True
		except Exception as e:
			raise e

	def collect(self, fake_loss=False):
		if not fake_loss:
			try:
				path = base_folder+'/result_pool/{}.npz'.format(self.scope.replace('/', '_'))
				result = np.load(path)
				self.repeat_loss = result['repeat_loss']
				os.remove(path)
				return True
			except Exception:
				return False
		else:
			self.repeat_loss = [9999]*self.repeat
			return True

class Generation(object):
	def __init__(self, n_generation,fitness_scores,history_populations,algo_code,pG=None, name=None, **kwargs):
		super(Generation, self).__init__()
		self.name = name
		self.N_islands = kwargs['N_islands'] if 'N_islands' in kwargs.keys() else 1
		self.kwargs = kwargs
		self.out = self.kwargs['out']
		self.init_seed = self.kwargs['Init_seed']
		self.rank = self.kwargs['rank']
		self.init_rank = self.kwargs['init_rank']
		self.size = self.kwargs['size']
		self.init_sparsity_LB = kwargs['init_sparsity_LB'] if 'init_sparsity_LB' in kwargs.keys() else 0.8
		self.indv_to_collect = []
		self.indv_to_distribute = []

		self.algo_code=algo_code
		self.history_populations=history_populations
		self.fitness_scores=fitness_scores
		self.best_fitness_scores=100000000
		self.n_generation = n_generation 

		if pG is not None:
			self.societies = {}
			for k, v in pG.societies.items():
				self.societies[k] = {}
				self.societies[k]['indv'] = \
						[ Individual( adj_matrix=indv.adj_matrix, parents=indv.parents,
													scope='{}/{}/{:03d}'.format(self.name, k, idx), **self.kwargs) \
						for idx, indv in enumerate(v['indv']) ]
				self.indv_to_distribute += [indv for indv in self.societies[k]['indv']]

		elif 'random_init' in kwargs.keys():
			self.societies = {}
			for n in range(self.kwargs['N_islands']):
				society_name = ''.join(choice(string.ascii_uppercase + string.digits) for _ in range(6))
				self.societies[society_name] = {}
				self.societies[society_name]['indv'] = [ \
						Individual(scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
						adj_matrix=self.__random_adj_matrix__(i), **self.kwargs) \
						for i in range(self.kwargs['population'][n]) ]
				self.indv_to_distribute += [indv for indv in self.societies[society_name]['indv']]

	def __call__(self, **kwargs):
		try:
			self.__evaluate__()
			if 'callbacks' in kwargs.keys():
				for c in kwargs['callbacks']:
					c(self)
			self.__evolve__()
			return True
		except Exception as e:
			raise e

	def __random_adj_matrix__(self,i,**kwargs):
		if isinstance(self.out, list):
			adj_matrix = np.diag(self.out)
		else:
			adj_matrix = np.diag(self.out)

		rng = np.random.default_rng(seed=(self.init_seed+i))

		sparsity_level=rng.uniform(self.init_sparsity_LB, 1)

		connection = [ int(rng.uniform()>sparsity_level)*rng.integers(low=2, high=(self.init_rank)) for i in range(np.sum(np.arange(self.size)))]


		connection = [1 if x == 0 else x for x in connection]
		adj_matrix[np.triu_indices(self.size, 1)] = connection


		## get the code save in history_populations
		# Get the indices of the upper triangle (excluding the main diagonal)
		rows, cols = np.triu_indices_from(adj_matrix, k=1)
		# Extract the elements into a one-dimensional array and convert to int
		upper_diagonal_vector = adj_matrix[rows, cols].astype(int)
		self.history_populations['1'].append(upper_diagonal_vector)

		return adj_matrix

	def __evolve__(self):

		def new_individuals_generator(island,generation):
				### preprocessing State


				self.fitness_scores.update({'{}'.format(generation):[]})
				for i in range(len(island['total'])):
								self.fitness_scores['{}'.format(generation)].append(float(island['total'][i]))

				### get the best individuls
				best_individual_fitness_scores=10000000000
				for i in range(len(self.history_populations)):
								for j in range(len(self.history_populations['{}'.format(i+1)])):
												if self.fitness_scores['{}'.format(i+1)][j]<best_individual_fitness_scores:
														best_individual_fitness_scores=self.fitness_scores['{}'.format(i+1)][j]
														self.best_fitness_scores=best_individual_fitness_scores
														best_iteration=i+1
														best_individual=self.history_populations['{}'.format(i+1)][j].copy()

				history_populations=self.history_populations.copy()
				fitness_scores=self.fitness_scores.copy()
# 				best_individual=1
				new_individuals_numbers=self.kwargs['population'][0]
				current_iteration=generation+1
				maximum_iteration=self.kwargs['max_generation2']
				### a dictionary of hyperparameters
				hyperparameters={'test':1}
				hyperparameters.update({'code_length': np.sum(np.arange(self.size))})
				### add indent befer running code of string
				added_indent='				'


				original_string=self.algo_code
				# Separator
				separator = 'def GenerateSample('

				# Splitting the string
				parts = original_string.split(separator, 1)

				# Extracting the parts
				if len(parts) == 1:
				    before = parts[0]
				    after = ""
				else:
				    before = parts[0]
				    after = separator + parts[1]



				already_import=['from openai import OpenAI\n','import re\n','import numpy as np\n','from random import choices\n','import pickle\n','import textwrap\n']

				substrings = already_import

				# Input string
				input_string = before

				# Remove each substring from the input string
				for substring in substrings:
				    input_string = input_string.replace(substring, '')

				before=input_string

				# Original string
				original_string = after

				# Word to add
				word_to_add = added_indent

				# Splitting the string into lines
				lines = original_string.split('\n')

				# Adding the word to the start of each line
				for i in range(len(lines)):
				    lines[i] = word_to_add + lines[i]

				# Joining the lines back together
				modified_string = '\n'.join(lines)





				def add_indentation_to_string(s, indentation):
				    # Split the string into lines
				    lines = s.split('\n')

				    # Strip leading spaces and add the desired indentation
				    modified_lines = [indentation + line.lstrip() for line in lines]

				    # Join the lines back together
				    modified_string = '\n'.join(modified_lines)

				    return modified_string

				# Example usage
				original_string = before
				original_string = textwrap.dedent(original_string)
				tnp_string = add_indentation_to_string(original_string,'')


				def visualize_string(s):
								return s.replace(" ", "<space>").replace("\t", "<tab>")

				visualized_string = visualize_string(tnp_string)



				exec(tnp_string, globals())


				function_code = textwrap.dedent(modified_string)
				function_code=before+function_code
				local_vars = {}
				globals_vars = globals()  # Get the global namespace

				# Execute the entire string to define all functions
				exec(function_code, globals_vars, local_vars)

				# Retrieve the 'GenerateSample' function
				GenerateSample = local_vars['GenerateSample']




				hyperparameters.update({'code_bound': self.kwargs['rank']})
				hyperparameters.update({'code_upperbound': self.kwargs['rank']})
				try:
						new_individuals=GenerateSample(history_populations,fitness_scores,best_individual,new_individuals_numbers,current_iteration,maximum_iteration,hyperparameters)
				except Exception as e:
						logging.info("An error occurred: %s", e)




				del GenerateSample
				###### post-processing after the generation of new_individuals 
				###### ensure new_individuals are legal adj code
				def is_vector_legal(vector, n, C):
						# Check if the length of the vector is n
						if len(vector) != n:
												return False
						# Check if all elements are integers
						if not issubclass(vector.dtype.type, np.integer):
												return False
						# Check if all elements are within the range [1, C]
						if np.any(vector < 1) or np.any(vector > C):
												return False
						return True

				def process_vector(vector, n, C):
						# Truncate or pad the vector to length n
						if len(vector) > n:
												vector = vector[:n]
						elif len(vector) < n:
						# Pad the vector with 1s (or any other value within [1, C])
												vector = np.pad(vector, (0, n - len(vector)), 'constant', constant_values=1)
						# Convert elements to integers
						vector = vector.astype(int)
						# Ensure elements are within [1, C]
						vector = np.clip(vector, 1, C)
						return vector


				# change to vector

				for i in range(len(new_individuals)):
						new_individuals[i]=np.array(new_individuals[i])
						new_individuals[i]=new_individuals[i].reshape(-1,order="F")

				# Example numpy vector
				legal_individuals=[]
				for i in range(len(new_individuals)):
						vector = new_individuals[i].copy()
						# Convert elements to integers
						vector = vector.astype(int)

						# Define the range [1, n]
						n = self.kwargs['rank']  # You can set n to your desired upper limit
						vector = np.clip(vector, 1, n)
						legal_individuals.append(vector)
				for i in range(len(legal_individuals)):
						if not is_vector_legal(legal_individuals[i], int(np.sum(np.arange(self.kwargs['size']))),  self.kwargs['rank']):
							logging.info('Something wrong with the adj_code!!!')
							legal_individuals[i]=process_vector(legal_individuals[i],int(np.sum(np.arange(self.kwargs['size']))),  self.kwargs['rank'])

				adjmat_set=[]
				for i in range(len(legal_individuals)):
						adj_matrix = np.diag(self.out)
						adj_matrix[np.triu_indices(self.size, 1)] =legal_individuals[i].tolist()
						adjmat_set.append(adj_matrix)




				if len(adjmat_set)>self.kwargs['population'][0]:
						for i in range(len(adjmat_set)-self.kwargs['population'][0]):
								adjmat_set.pop()


				self.history_populations.update({'{}'.format(current_iteration):[]})

				for i in range(len(adjmat_set)):
						rows, cols = np.triu_indices_from(adj_matrix, k=1)
						# Extract the elements into a one-dimensional array and convert to int
						upper_diagonal_vector = adjmat_set[i][rows, cols].astype(int)
						self.history_populations['{}'.format(current_iteration)].append(upper_diagonal_vector)




				if len(adjmat_set)<len(island['indv']):
							for i in range(int(len(island['indv'])-len(adjmat_set))): 
											island['indv'].pop()
							for i in range(len(island['indv'])): 
											island['indv'][i].adj_matrix=adjmat_set[i]

											island['indv'][i].parents=('%d'%(best_iteration))

				if len(adjmat_set)>len(island['indv']):
							for i in range(int(len(adjmat_set)-len(island['indv']))): 
											island['indv'].append(DummyIndv())
							for i in range(len(island['indv'])): 
											island['indv'][i].adj_matrix=adjmat_set[i]

											island['indv'][i].parents=('%d'%(best_iteration))

				if len(adjmat_set)==len(island['indv']):
							for i in range(len(island['indv'])): 
											island['indv'][i].adj_matrix=adjmat_set[i]

											island['indv'][i].parents=('%d'%(best_iteration))


				

		for idx, (k, v) in enumerate(self.societies.items()):
				new_individuals_generator(v,self.n_generation)


	def __evaluate__(self):

		def score2rank(island, idx):
			sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))
			score = island['score']
			sparsity_score = [ s for s, l in score ]
			loss_score = [ l for s, l in score ]

			if 'fitness_func' in self.kwargs.keys():
				if isinstance(self.kwargs['fitness_func'], list):
					fitness_func = self.kwargs['fitness_func'][idx]
				else:
					fitness_func = self.kwargs['fitness_func']
			else:		
				fitness_func = lambda s, l: 1*s+200*l
			
			total_score = [ fitness_func(s, l) for s, l in zip(sparsity_score, loss_score) ]

			island['rank'] = np.argsort(total_score)
			island['total'] = total_score

		# RANKING
		for idx, (k, v) in enumerate(self.societies.items()):
			v['score'] = [ (indv.sparsity ,np.min(indv.repeat_loss)) for indv in v['indv'] ]
			score2rank(v, idx)

	def distribute_indv(self, agent):
		if self.indv_to_distribute:
			indv = self.indv_to_distribute.pop(0)
			if np.log10(indv.sparsity)<-0.7372986845760743:
				agent.receive(indv)
				self.indv_to_collect.append(indv)
				logging.info('Assigned individual {} to agent {}.'.format(indv.scope, agent.sge_job_id))
			else:
				indv.collect(fake_loss=True)
				logging.info('Individual {} is killed due to its sparsity = {} / {}.'.format(indv.scope, np.log10(indv.sparsity), indv.sparsityB))

	def collect_indv(self):
		for indv in self.indv_to_collect:
			if indv.collect():
				logging.info('Collected individual result {}.'.format(indv.scope))
				self.indv_to_collect.remove(indv)

	def is_finished(self):
		if len(self.indv_to_distribute) == 0 and len(self.indv_to_collect) == 0:
			return True
		else:
			return False

	def get_center(self):
			return self.best_fitness_scores,self.fitness_scores,self.history_populations


class Agent(object):
	def __init__(self, **kwargs):
		super(Agent, self).__init__()
		self.kwargs = kwargs
		self.sge_job_id = self.kwargs['sge_job_id']

	def receive(self, indv):
		indv.deploy(self.sge_job_id)
		with open(base_folder+'/agent_pool/{}.POOL'.format(self.sge_job_id), 'a') as f:
			f.write(evoluation_goal)

	def is_available(self):
		return True if os.stat(base_folder+'/agent_pool/{}.POOL'.format(self.kwargs['sge_job_id'])).st_size == 0 else False

class Overlord(object):
	def __init__(self, max_generation=100, **kwargs):
		super(Overlord, self).__init__()
		self.dummy_func = lambda *args, **kwargs: None
		self.max_generation = max_generation
		self.current_generation = None
		self.previous_generation = None
		self.N_generation = 0
		self.kwargs = kwargs
		self.generation = kwargs['generation']
		self.generation_list = []
		self.available_agents = []
		self.known_agents = {}
		self.time = 0
		self.AS_iteration = kwargs['AS_iteration']
		self.train_data_list =kwargs['train_data_list']
		self.folder_name =kwargs['run_data_type']
		self.repeat_runnning =kwargs['repeat_running']
		self.algo_clusters=kwargs['algo_clusters']

		self.gpt_model=kwargs['gpt_model']

		self.m_incontext=kwargs['m_incontext']
		self.n_crossover=kwargs['n_crossover']


		# Check if the folder exists before creating it
		if not os.path.exists('./'+self.folder_name):
				os.mkdir(self.folder_name)

		self.destination_log_path = './'+self.folder_name+'/structure_searching_results.log'


	def __call_with_interval__(self, func, interval):
		return func if self.time%interval == 0 else self.dummy_func

	def __tik__(self, sec):
		# logging.info(self.time)
		self.time += sec
		time.sleep(sec)

	def __check_available_agent__(self):
		self.available_agents.clear()
		agents = glob.glob(base_folder+'/agent_pool/*.POOL')
		agents_id = [ a.split('/')[-1][:-5] for a in agents ]

		for aid in list(self.known_agents.keys()):
			if aid not in agents_id:
				logging.info('Dead agent id = {} found!'.format(aid))
				self.known_agents.pop(aid, None)

		for aid in agents_id:
			if aid in self.known_agents.keys():
				if self.known_agents[aid].is_available():
					self.available_agents.append(self.known_agents[aid])
			else:
				self.known_agents[aid] = Agent(sge_job_id=aid)
				logging.info('New agent id = {} found!'.format(aid))

	def __assign_job__(self):
		self.__check_available_agent__()
		if len(self.available_agents)>0:
			for agent in self.available_agents:
				self.current_generation.distribute_indv(agent)

	def __collect_result__(self):
		self.current_generation.collect_indv()

	def __report_agents__(self):
		logging.info('Current number of known agents is {}.'.format(len(self.known_agents)))
		logging.info(list(self.known_agents.keys()))

	def __report_generation__(self):
		logging.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
		logging.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
		logging.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])

	def __generation__(self,algo_code):
		if self.N_generation > self.max_generation:
			return False
		else:
			if self.current_generation is None:
				history_populations={}
				history_populations.update({'1':[]})
				fitness_scores={}
				self.current_generation = self.generation(0,fitness_scores,history_populations,algo_code,name='generation_init', **self.kwargs)
				self.current_generation.indv_to_distribute = []

			if self.current_generation.is_finished():
				if self.previous_generation is not None:
					self.current_generation(**self.kwargs)
					fitness,fitness_scores,history_populations=self.previous_generation.get_center()

				self.N_generation += 1
				self.previous_generation = self.current_generation
				self.current_generation = self.generation(self.N_generation,fitness_scores,history_populations,algo_code,self.previous_generation, 
														name='generation_{:03d}'.format(self.N_generation), **self.kwargs)

			return True

	def __call__(self):



		algorithms_search_process={}
# 		file_path = "./my_algorithms_searching.pkl"
		file_path="./"+self.folder_name +"/my_algorithms_searching.pkl"
		# Save the dictionary to a JSON file
		with open(file_path, 'wb') as pickle_file:
				pickle.dump(algorithms_search_process, pickle_file)
		

		init_algorithms_pool_V2(self.folder_name)
# 		file_path = "./my_algorithms_pool.pkl"
		file_path="./"+self.folder_name+"/my_algorithms_pool.pkl"

		# Read the dictionary from the binary file
		with open(file_path, 'rb') as pickle_file:
 		         populations_of_algorithms = pickle.load(pickle_file)


		for i in range(len(populations_of_algorithms)):
						novel_alg_code=populations_of_algorithms['cluster{}'.format(i+1)]['code_alg1']
						try:
							fitness_training=[]
							for j in range(len(self.train_data_list)):
								fitness_per_image=[]
								for k in range(self.repeat_runnning):
									self.N_generation = 0
									self.current_generation = None
									self.previous_generation = None
									data_name=self.train_data_list[j]
									data = np.load(data_name)
									logging.info(data['adj_matrix'])
									adjm=data['adj_matrix']
									adjm[adjm==0] = 1

									actual_elem = np.sum([ np.prod(adjm[d]) for d in range(adjm.shape[0]) ])
									logging.info(actual_elem)

									np.save('data.npy', data['goal'])

									while self.__generation__(novel_alg_code):
										self.__call_with_interval__(self.__check_available_agent__, 4)()
										self.__call_with_interval__(self.__assign_job__, 4)()
										self.__call_with_interval__(self.__collect_result__, 4)()
										self.__call_with_interval__(self.__report_agents__, 180)()
										self.__call_with_interval__(self.__report_generation__, 160)()
										self.__tik__(2)
									#### use the self.previous_generation.get_center() to get the fitness_score
									fitness,_,_=self.previous_generation.get_center()
									fitness_per_image.append(fitness)
									logging.info(fitness)
								fitness_training.append(np.average(fitness_per_image))
						except Exception as e:
							fitness_training=[1000000]



						populations_of_algorithms['cluster{}'.format(i+1)]['fit_alg1']=sum(fitness_training) / len(fitness_training)
						logging.info("Above is the results for method at cluster{}-algorithm{}".format(i+1,1))
						shutil.copyfile(source_log_path, self.destination_log_path)
						with open(file_path, 'wb') as pickle_file:
								pickle.dump(populations_of_algorithms, pickle_file)





				# Done initialze the algorithms pool
		for iteration_number in range(self.AS_iteration):
						recommend_algorithm=function_crossover(self.folder_name,self.gpt_model,self.m_incontext,self.n_crossover)
						if recommend_algorithm[0]==True:
								break
						else:
								recommend_algorithm=function_mutation(recommend_algorithm,self.gpt_model)

								fitness_algorithms=[]
								for algorithm_index in range(len(recommend_algorithm)-1):
									novel_alg_code=recommend_algorithm[algorithm_index+1]['alg_code']
									try:
										fitness_training=[]
										for j in range(len(self.train_data_list)):
											fitness_per_image=[]
											for k in range(self.repeat_runnning):
												self.N_generation = 0
												self.current_generation = None
												self.previous_generation = None
												data_name=self.train_data_list[j]
												data = np.load(data_name)
												logging.info(data['adj_matrix'])
												adjm=data['adj_matrix']
												adjm[adjm==0] = 1

												actual_elem = np.sum([ np.prod(adjm[d]) for d in range(adjm.shape[0]) ])
												logging.info(actual_elem)

												np.save('data.npy', data['goal'])

												while self.__generation__(novel_alg_code):
													self.__call_with_interval__(self.__check_available_agent__, 4)()
													self.__call_with_interval__(self.__assign_job__, 4)()
													self.__call_with_interval__(self.__collect_result__, 4)()
													self.__call_with_interval__(self.__report_agents__, 180)()
													self.__call_with_interval__(self.__report_generation__, 160)()
													self.__tik__(2)
												#### use the self.previous_generation.get_center() to get the fitness_score
												fitness,_,_=self.previous_generation.get_center()
												fitness_per_image.append(fitness)
												logging.info(fitness)
											fitness_training.append(np.average(fitness_per_image))
									except Exception as e:
										fitness_training=[1000000]
									fitness_algorithms.append(fitness_training)


								for algorithm_index in range(len(recommend_algorithm)-1):
										recommend_algorithm[algorithm_index+1].update({'alg_fitness':sum(fitness_algorithms[algorithm_index]) / len(fitness_algorithms[algorithm_index])})


								goto_cluster, goto_cluster_index=Post_processing(recommend_algorithm,self.folder_name,self.gpt_model)
								logging.info("Above is the reuslts for method at cluster{}-algorithm{}".format(goto_cluster,goto_cluster_index))
								shutil.copyfile(source_log_path, self.destination_log_path)
                        ### generate novel cluster
                        ### check if generated sucessfully

								# file_path = "./my_algorithms_pool.pkl"
								file_path="./"+self.folder_name+"/my_algorithms_pool.pkl"
								# Read the dictionary from the binary file
								with open(file_path, 'rb') as pickle_file:
										populations_of_algorithms = pickle.load(pickle_file)
                        
								cluster_number_old=len(populations_of_algorithms)
								if cluster_number_old<self.algo_clusters:



										
										
										generate_novel_cluster(self.folder_name,self.gpt_model)
		                        
		                        ### check if generated sucessfully
		
										# file_path = "./my_algorithms_pool.pkl"
										file_path="./"+self.folder_name+"/my_algorithms_pool.pkl"
										# Read the dictionary from the binary file
										with open(file_path, 'rb') as pickle_file:
												populations_of_algorithms = pickle.load(pickle_file)
		                        
										cluster_number_new=len(populations_of_algorithms)
										if cluster_number_new!=cluster_number_old:
		                            
												recommend_algorithm={}
												recommend_algorithm.update({'alg_descri':''})
												recommend_algorithm.update({'alg_code':populations_of_algorithms['cluster{}'.format(len(populations_of_algorithms))]['code_alg1']})


												populations_of_algorithms['cluster{}'.format(len(populations_of_algorithms))]['code_alg1']=recommend_algorithm['alg_code']
												populations_of_algorithms['cluster{}'.format(len(populations_of_algorithms))]['descri_alg1']=recommend_algorithm['alg_descri']
												# file_path = "./my_algorithms_pool.pkl"
												file_path="./"+self.folder_name+"/my_algorithms_pool.pkl"
												with open(file_path, 'wb') as pickle_file:
														pickle.dump(populations_of_algorithms, pickle_file)
		
		
												novel_alg_code=populations_of_algorithms['cluster{}'.format(len(populations_of_algorithms))]['code_alg1']

		
												try:
													fitness_training=[]
													for j in range(len(self.train_data_list)):
														fitness_per_image=[]
														for k in range(self.repeat_runnning):
															self.N_generation = 0
															self.current_generation = None
															self.previous_generation = None
															data_name=self.train_data_list[j]
															data = np.load(data_name)
															logging.info(data['adj_matrix'])
															adjm=data['adj_matrix']
															adjm[adjm==0] = 1
		
															actual_elem = np.sum([ np.prod(adjm[d]) for d in range(adjm.shape[0]) ])
															logging.info(actual_elem)
		
															np.save('data.npy', data['goal'])
		
															while self.__generation__(novel_alg_code):
																self.__call_with_interval__(self.__check_available_agent__, 4)()
																self.__call_with_interval__(self.__assign_job__, 4)()
																self.__call_with_interval__(self.__collect_result__, 4)()
																self.__call_with_interval__(self.__report_agents__, 180)()
																self.__call_with_interval__(self.__report_generation__, 160)()
																self.__tik__(2)
															#### use the self.previous_generation.get_center() to get the fitness_score
															fitness,_,_=self.previous_generation.get_center()
															fitness_per_image.append(fitness)
															logging.info(fitness)
														fitness_training.append(np.average(fitness_per_image))
												except Exception as e:
													fitness_training=[1000000]

		
												recommend_algorithm.update({'alg_fitness':sum(fitness_training) / len(fitness_training)})


												populations_of_algorithms['cluster{}'.format(len(populations_of_algorithms))]['code_alg1']=recommend_algorithm['alg_code']
												populations_of_algorithms['cluster{}'.format(len(populations_of_algorithms))]['descri_alg1']=recommend_algorithm['alg_descri']
												populations_of_algorithms['cluster{}'.format(len(populations_of_algorithms))]['fit_alg1']=recommend_algorithm['alg_fitness']
												logging.info("Above is the reuslts for method at cluster{}-algorithm{}".format(len(populations_of_algorithms),1))
												shutil.copyfile(source_log_path, self.destination_log_path)
												# file_path = "./my_algorithms_pool.pkl"
												file_path="./"+self.folder_name+"/my_algorithms_pool.pkl"
												with open(file_path, 'wb') as pickle_file:
														pickle.dump(populations_of_algorithms, pickle_file)





def score_summary(obj):
	logging.info('===== {} ====='.format(obj.name))

	for k, v in obj.societies.items():
		logging.info('===== ISLAND {} ====='.format(k))

		for idx, indv in enumerate(v['indv']):
			if idx == v['rank'][0]:
				logging.info('\033[31m{} | {:.3f} | {} | {:.5f} | {}|{}\033[0m'.format(indv.scope, np.log10(indv.sparsity), [ float('{:.14f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents,indv.actual_elements))
				adj_matrix_in=indv.adj_matrix.copy()
				adj_matrix_in[adj_matrix_in==1] = 0
				logging.info(adj_matrix_in)
			else:
				logging.info('{} | {:.3f} | {} | {:.5f} | {}|{}'.format(indv.scope, np.log10(indv.sparsity), [ float('{:.14f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents,indv.actual_elements))
				adj_matrix_in=indv.adj_matrix.copy()
				
				adj_matrix_in[adj_matrix_in==1] = 0
				logging.info(adj_matrix_in)

if __name__ == '__main__':
	pipeline = Overlord(		# GENERATION PROPERTIES
													max_generation=20,generation=Generation, random_init=True,AS_iteration=50,Init_seed=1234,
													# ISLAND PROPERTIES
													train_data_list=['live_0_8D.npz','live_3_8D.npz','live_5_8D.npz','live_6_8D.npz'],repeat_running=1,run_data_type='large_scale',m_incontext=2,n_crossover=1,
													N_islands=1, population=[int(sys.argv[1])], max_generation2=20,algo_clusters=5,
													# INVIDUAL PROPERTIES
													size=8, rank=4, out=np.array([4,4,4,4,4,4,4,4],dtype=int), init_sparsity_LB=0.85,init_rank=3,gpt_model='gpt-4-1106-preview',
													evaluate_repeat=1, max_iterations=40000,
													fitness_func=[ lambda s, l: s+5*l],
													rse_therhold=[1e-8],Adam_Step=[0.01],

													# FOR COMPUTATION
													callbacks=[score_summary])
	pipeline()