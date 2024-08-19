from openai import OpenAI
import re
import numpy as np
from random import choices
import pickle
import textwrap


def init_algorithms_pool_V2(file_name):
    populations_of_algorithms={}
    populations_of_algorithms.update({'cluster{}'.format(1):{}})
    populations_of_algorithms.update({'cluster{}'.format(2):{}})
    populations_of_algorithms.update({'cluster{}'.format(3):{}})

    ### Initial algorithm with fake fitness scores
    populations_of_algorithms['cluster1'].update({'code_alg1':"def GenerateSample(history_populations,fitness_scores,best_individual,new_individuals_numbers,current_iteration,maximum_iteration,hyperparameters):\n    Ranking=np.argsort(fitness_scores['{}'.format(current_iteration-1)])\n    elite_num=int(len(fitness_scores['{}'.format(current_iteration-1)])*hyperparameters.get('elimination_percentage', 0.9))\n    Ranking=Ranking[0:elite_num]\n    populations_elite=[history_populations['{}'.format(current_iteration-1)][i].copy() for i in Ranking]\n    fitness_scores_elite=[fitness_scores['{}'.format(current_iteration-1)][i] for i in Ranking]\n    Rank_elite = np.argsort(fitness_scores_elite)\n    p = [ np.maximum(np.log(hyperparameters.get('alpha', 100)/(0.01+k*5)), 0.01) for k in range(len(populations_elite)) ]\n    prob = np.zeros(len(populations_elite))\n    for idx, i in enumerate(Rank_elite): prob[i] = p[idx]\n    new_individuals=[]\n    for i in range(new_individuals_numbers//2): \n        parents=choices(populations_elite, weights=prob, k=2)\n        female=parents[0].copy()\n        male=parents[1].copy()\n        index=np.arange(len(male))\n        np.random.shuffle(index)\n        index=index[0:(len(male)//2)]\n        tnp=female[index]\n        female[index]=male[index]\n        male[index]=tnp\n        new_individuals.append(male)\n        new_individuals.append(female)\n    if np.mod(new_individuals_numbers,2)!=0:\n        tnp=new_individuals[-1].copy()\n        np.random.shuffle(tnp)\n        new_individuals.append(tnp)\n    for i in range(new_individuals_numbers):\n        mask = np.random.uniform(0,1,[len(new_individuals[0])])<hyperparameters.get('mutation_rate', 0.25)\n        for j in range(len(new_individuals[0])):\n            if mask[j]:\n                mutate_range=np.arange(1,hyperparameters.get('code_upperbound', 15)+1)\n                mutate_range=np.delete(mutate_range, np.where(mutate_range == new_individuals[i][j]))\n                np.random.shuffle(mutate_range)\n                new_individuals[i][j]=mutate_range[0]\n    return new_individuals"})
    populations_of_algorithms['cluster1'].update({'descri_alg1':'The function sample for "GenerateSample" uses ranking, selection, crossover, and mutation for generating new individuals.'})
    populations_of_algorithms['cluster1'].update({'fit_alg1':100000})

    populations_of_algorithms['cluster2'].update({'code_alg1':"def GenerateSample(history_populations,fitness_scores,best_individual,new_individuals_numbers,current_iteration,maximum_iteration,hyperparameters):\n     variance=hyperparameters.get('decay_rate', 0.99)**(current_iteration-2)\n     if variance<hyperparameters.get('variance_LB', 0.3):\n         variance=hyperparameters.get('variance_LB', 0.3)\n     new_individuals=[]\n     for i in range(new_individuals_numbers):\n         tnp=np.array(best_individual)+np.random.randn(len(best_individual))*variance\n         tnp=np.round(tnp)\n         tnp[np.where(tnp>hyperparameters.get('code_upperbound', 15))]=hyperparameters.get('code_upperbound', 15)\n         tnp[np.where(tnp<1)]=1\n         tnp=tnp.astype(int)\n         new_individuals.append(tnp)\n     return new_individuals"})
    populations_of_algorithms['cluster2'].update({'descri_alg1':'The function generates new individuals by adding normally distributed random values, scaled by a decayed variance, to the best individual.'})
    populations_of_algorithms['cluster2'].update({'fit_alg1':100000})


    populations_of_algorithms['cluster3'].update({'code_alg1':"def GenerateSample(history_populations,fitness_scores,best_individual,new_individuals_numbers,current_iteration,maximum_iteration,hyperparameters):\n    new_individuals=[]\n    for i in range(len(best_individual)):\n        if best_individual[i]<hyperparameters.get('code_upperbound', 15):\n            tnp=best_individual.copy()\n            tnp[i]=tnp[i]+1\n            new_individuals.append(tnp)\n    for i in range(len(best_individual)):\n        if best_individual[i]>1:\n            tnp=best_individual.copy()\n            tnp[i]=tnp[i]-1\n            new_individuals.append(tnp)\n    return new_individuals"})
    populations_of_algorithms['cluster3'].update({'descri_alg1':'The function creates variations of the best individual by incrementally increasing and decreasing each element within bounds.'})
    populations_of_algorithms['cluster3'].update({'fit_alg1':100000})


    # file_path = "./my_algorithms_pool.pkl"
    file_path="./"+file_name+"/my_algorithms_pool.pkl"
    # Save the dictionary to a JSON file
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(populations_of_algorithms, pickle_file)



def function_crossover(file_name,gpt_model,m_incontext,n_crossover):

    # file_path = "./my_algorithms_pool.pkl"
    file_path="./"+file_name+"/my_algorithms_pool.pkl"
# Read the dictionary from the binary file
    with open(file_path, 'rb') as pickle_file:
        populations_of_algorithms = pickle.load(pickle_file)


    cluster_num=len(populations_of_algorithms)
    clusters_size=np.arange(cluster_num)
    clusters_fit=[]
    clusters_best_fit=[]
    clusters_best_fit_algo_num=[]
    for i in range(cluster_num):
            clusters_size[i]=int(len(populations_of_algorithms['cluster{}'.format(i+1)])//3)
            tnp=[]
            for j in range(int(len(populations_of_algorithms['cluster{}'.format(i+1)])//3)):
                tnp.append(populations_of_algorithms['cluster{}'.format(i+1)]['fit_alg{}'.format(j+1)])
            clusters_fit.append(tnp.copy())
            clusters_best_fit.append(min(tnp))
            clusters_best_fit_algo_num.append(tnp.index(min(tnp))+1)

    if m_incontext>int(np.sum(clusters_size)):
        m_incontext=int(np.sum(clusters_size))


    # file_path = "./my_algorithms_searching.pkl"
    file_path="./"+file_name+"/my_algorithms_searching.pkl"
# Read the dictionary from the binary file
    with open(file_path, 'rb') as pickle_file:
        algorithms_search_process = pickle.load(pickle_file)

    function_description="Function description:\ndef GenerateSample(history_populations,fitness_scores,best_individual,new_individuals_numbers,current_iteration,maximum_iteration,hyperparameters):\n     ## GenerateSample: Function takes in integer vectors and output integer vectors.\n     ## Input: \n     ## history_populations: Dictionary. Keys are integer strings from '1' to some larger value. Each key contains a list of integer numpy vectors.\n     ## fitness_scores: Dictionary. Keys are same as history_populations. Each key contains a list of floats, the lower the better.\n     ## best_individual: Numpy integer vector.\n     ## new_individuals_numbers: Integer.\n     ## current_iteration: Integer, 1 larger than len of history_populations.\n     ## maximum_iteration: Integer.\n     ## hyperparameters: Dictionary. Keys are strings contain the constant used for computation. Default values should be provided using .get().\n     ## Output: \n     ## new_individuals: List, len new_individuals_numbers, contains integer numpy vectors. Each vector's len is the same as the len of the vectors in history_populations. Furthermore, The elements are within range [1,hyperparameters['code_upperbound']].\n     return new_individuals\n"

    instruction="Functions 1 to {} are implementations of the ‘GenerateSample’. A Lower score implies better function. Learning from their results, think about what works and what doesn’t, provide {} novel methods with lower score. You are encouraged to be creative to incorporate novel ideas but do not simply stack methods together. ".format(m_incontext,n_crossover)
    if n_crossover==1:
        output="Provide runnable code that has implemented all your ideas (If any part requires choice, use choices from random). Do not leave any placeholder for me, all the functionality should be actually implemented. Your response format\n<code>(must include this sign)\nYour code\n</code>(must include this sign)\nAlso, you don’t need to add any other words.\n"

    else:
        output="Provide runnable code that has implemented all your ideas (If any part requires choice, use choices from random). Do not leave any placeholder for me, all the functionality should be actually implemented. Your response format\n<code>(must include this sign)\nYour code 1\n</code>(must include this sign)\n...\n<code>(must include this sign)\nYour code {}\n</code>(must include this sign)\nAlso, you don’t need to add any other words and name each code 'GenerateSample'.\n".format(n_crossover)
    meta_prompt=instruction+output



    code_wrong_flag=True
    wrong_flag=True
    exec_wrong_flag=True
    find_no_code_flag=True

    LLM_wrong_flag=True
    re_prompt_num=0
    while (re_prompt_num<5)&((code_wrong_flag==True) | (wrong_flag==True)| (exec_wrong_flag==True)| (find_no_code_flag==True)|  (LLM_wrong_flag==True)):
        code_wrong_flag=False
        wrong_flag=False
        exec_wrong_flag=False
        find_no_code_flag=False

        LLM_wrong_flag=False
        re_prompt_num=re_prompt_num+1
        cluster_num=len(populations_of_algorithms)
        clusters_size=np.arange(cluster_num)
        clusters_fit=[]
        clusters_best_fit=[]
        clusters_best_fit_algo_num=[]
        for i in range(cluster_num):
            clusters_size[i]=int(len(populations_of_algorithms['cluster{}'.format(i+1)])//3)
            tnp=[]
            for j in range(int(len(populations_of_algorithms['cluster{}'.format(i+1)])//3)):
                tnp.append(populations_of_algorithms['cluster{}'.format(i+1)]['fit_alg{}'.format(j+1)])
            clusters_fit.append(tnp.copy())
            clusters_best_fit.append(min(tnp))
            clusters_best_fit_algo_num.append(tnp.index(min(tnp))+1)
        ### CREATE POBABILITY AND CREAT SELECTION INDEX

        Rank = np.argsort(np.array(clusters_best_fit))
        p = [ np.maximum(np.log(100/(0.01+k*5)), 0.01) for k in range(cluster_num) ]
        prob = np.zeros(cluster_num)
        for idx, i in enumerate(Rank): prob[i] = p[idx]
        ### SAMPLE THE CLUSTER INDEX FIRST
        clusters_size_tnp=clusters_size.copy()
        index_cluster=(np.arange(cluster_num)+1).tolist()
        select_cluster_index=[]
        for i in range(m_incontext):
            if len(index_cluster)==0:
                break
            select_cluster_index.append(choices(index_cluster, weights=prob, k=1)[0])
            clusters_size_tnp[select_cluster_index[-1]-1]=clusters_size_tnp[select_cluster_index[-1]-1]-1
            if clusters_size_tnp[select_cluster_index[-1]-1]==0:
                index_delete=index_cluster.index(select_cluster_index[-1])
                prob = np.delete(prob, index_delete)
                index_cluster.pop(index_delete)
        numbers = select_cluster_index
        # Using a dictionary to count occurrences
        counts = {}
        for num in numbers:
            if num in counts:
                counts[num] += 1
            else:
                counts[num] = 1
        ###GENRATE EACH INDIV INDEX
        def weighted_choice_without_replacement(items, weights, number_to_select):
            items_array = np.array(items)
            weights_array = np.array(weights)

            selected_indices = np.array([], dtype=int)

            for _ in range(number_to_select):
                normalized_weights = weights_array / weights_array.sum()
                chosen_index = np.random.choice(len(items_array), p=normalized_weights)
                selected_indices = np.append(selected_indices, chosen_index)

                # Set the weight of the chosen item to 0 to simulate removal
                weights_array[chosen_index] = 0

            return items_array[selected_indices]
        sample_cluster_index=[]
        sample_cluster_indivdual_index=[]
        for key, value in counts.items():
            sample_cluster_index.append(key)
            Rank = np.argsort(np.array(clusters_fit[key-1]))
            p = [ np.maximum(np.log(100/(0.01+k*5)), 0.01) for k in range(len(clusters_fit[key-1])) ]
            prob = np.zeros(len(clusters_fit[key-1]))
            algo_range=(np.arange(len(clusters_fit[key-1]))+1).tolist()
            for idx, i in enumerate(Rank): prob[i] = p[idx]
            prob=(prob/np.sum(prob)).tolist()
            
            sample_cluster_indivdual_index.append(weighted_choice_without_replacement(algo_range, prob, value).tolist())
        ###  CONSTRUCT ALGO PROMOPT FROM  THE SELECT ALGO



        tnp=""
        count=0
        for cluster_index in range(len(sample_cluster_index)):
            for cluster_indivdual_index in range(len(sample_cluster_indivdual_index[cluster_index])):
                count=count+1
                tnp=tnp+"Function {}:\n".format(count)
                tnp=tnp+populations_of_algorithms['cluster{}'.format(sample_cluster_index[cluster_index])]['code_alg{}'.format(sample_cluster_indivdual_index[cluster_index][cluster_indivdual_index])]+"\n"
                tnp=tnp+"Function {} score: {}\n".format(count,populations_of_algorithms['cluster{}'.format(sample_cluster_index[cluster_index])]['fit_alg{}'.format(sample_cluster_indivdual_index[cluster_index][cluster_indivdual_index])])
        algorithms_prompt=tnp



        client = OpenAI(

            api_key=open("./Key.txt", "r").read().strip("\n"),

            base_url="https://api.aiproxy.io/v1"
        )

        LLM_wrong_flag=False


        in_context_prompt=function_description+algorithms_prompt+meta_prompt

        try:
            completion = client.chat.completions.create(
              model=gpt_model, 
              temperature=0.7,
              messages=[{"role": "user", "content": in_context_prompt}]
            )
        except:
            LLM_wrong_flag=True


        if LLM_wrong_flag !=True:
            find_no_code_flag=False


            pattern = re.compile(r'<code>\n(.*?)\n</code>',re.DOTALL)
            match = pattern.findall(completion.choices[0].message.content)
            if match:
                    novel_alg_code_list=match
            else:
                    find_no_code_flag=True




            if (find_no_code_flag!=True):
              working_algorithms_index=[]
              problematical_codes=0
              for algorithm_index in range(len(novel_alg_code_list)):
                novel_alg_code=novel_alg_code_list[algorithm_index]
                ### add indent befer running code of string
                added_indent='            '

                original_string=novel_alg_code
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


                exec_wrong_flag=False

                try:
                         
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
                    exec(function_code, globals(), local_vars)
                    # Retrieve the defined function from local_vars
                    GenerateSample = local_vars['GenerateSample']
                except:
                    exec_wrong_flag=True
                if exec_wrong_flag!=True:
                    code_length=10
                    new_individuals_numbers=50
                    maximum_iteration=30
                    current_iteration=np.random.randint(2,maximum_iteration+1)
                    hyperparameters={}
                    history_populations={}
                    fitness_scores={}
                    hyperparameters.update({'code_bound':10})
                    hyperparameters.update({'code_upperbound': 10})
                    BF=1000000
                    code_wrong_flag=False
                    ## Generate individuals and their fitness scores
                    for i in range(current_iteration-1):
                        history_populations.update({'{}'.format(i+1):[]})
                        fitness_scores.update({'{}'.format(i+1):[]})
                        for j in range(np.random.randint(10,new_individuals_numbers+1)):
                            history_populations['{}'.format(i+1)].append(np.random.randint(1,hyperparameters['code_bound']+1,[code_length]))
                            fitness_scores['{}'.format(i+1)].append(float(np.abs(np.random.randn())))
                            if fitness_scores['{}'.format(i+1)][-1]<BF:
                                BF=fitness_scores['{}'.format(i+1)][-1]
                                best_individual=history_populations['{}'.format(i+1)][-1].copy()

                    try:
                        new_individuals=GenerateSample(history_populations, fitness_scores, best_individual, new_individuals_numbers, current_iteration, maximum_iteration, hyperparameters)
                    except:
                        code_wrong_flag=True
                    if code_wrong_flag !=True:
                        ## Check new_individuals
                        if code_wrong_flag==False:
                            wrong_flag=False
                            for i in range(len(new_individuals)):
                                tnp=new_individuals[i].copy()
                                tnp=np.array(tnp).reshape(-1)
                                if len(tnp)!=code_length:
                                    wrong_flag=True
                                    break
                        ## need to make sure all the 'wrong flag' are false 'code_wrong_flag', 'wrong_flag','exec_wrong_flag','find_no_code_flag','find_no_code_descri_flag','LLM_wrong_flag'
                if ((exec_wrong_flag==True)| (code_wrong_flag==True)| (wrong_flag==True))==True:
                                problematical_codes=problematical_codes+1
                                exec_wrong_flag=False
                                code_wrong_flag=False
                                wrong_flag=False
                else:
                                working_algorithms_index.append(algorithm_index)
            if len(working_algorithms_index)==0:
                wrong_flag=True
            else:
                exec_wrong_flag=False
                code_wrong_flag=False
                wrong_flag=False


    recommend_algorithm=[]

    if ((code_wrong_flag==True) | (wrong_flag==True)| (exec_wrong_flag==True)| (find_no_code_flag==True)| (LLM_wrong_flag==True))==False:
        
        len_tnp=len(algorithms_search_process)
        algorithms_search_process.update({'Step{}'.format(len_tnp+1):{}})

        
        for i in range(len(sample_cluster_index)):
            
            algorithms_search_process['Step{}'.format(len_tnp+1)].update({'Fromcluster{}'.format(sample_cluster_index[i]):sample_cluster_indivdual_index[i]})
        
        ### save the new population locally
        # file_path = "./my_algorithms_searching.pkl"
        file_path="./"+file_name+"/my_algorithms_searching.pkl"
        # Save the dictionary to a JSON file
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(algorithms_search_process, pickle_file)
        
        
        LLM_error=False
        recommend_algorithm.append(LLM_error)
        
        for i in range(len(working_algorithms_index)):
            novel_alg_code=novel_alg_code_list[working_algorithms_index[i]]
            algorithm={}
            algorithm.update({'alg_descri':''})
            algorithm.update({'alg_code':novel_alg_code})
            recommend_algorithm.append(algorithm)

    else:
        LLM_error=True
        recommend_algorithm.append(LLM_error)


    return recommend_algorithm


def function_mutation(recommend_algorithm,gpt_model):
    function_description="Function description:\ndef GenerateSample(history_populations,fitness_scores,best_individual,new_individuals_numbers,current_iteration,maximum_iteration,hyperparameters):\n     ## GenerateSample: Function takes in integer vectors and output integer vectors.\n     ## Input: \n     ## history_populations: Dictionary. Keys are integer strings from '1' to some larger value. Each key contains a list of integer numpy vectors.\n     ## fitness_scores: Dictionary. Keys are same as history_populations. Each key contains a list of floats, the lower the better.\n     ## best_individual: Numpy integer vector.\n     ## new_individuals_numbers: Integer.\n     ## current_iteration: Integer, 1 larger than len of history_populations.\n     ## maximum_iteration: Integer.\n     ## hyperparameters: Dictionary. Keys are strings contain the constant used for computation. Default values should be provided using .get().\n     ## Output: \n     ## new_individuals: List, len new_individuals_numbers, contains integer numpy vectors. Each vector's len is the same as the len of the vectors in history_populations. Furthermore, The elements are within range [1,hyperparameters['code_upperbound']].\n     return new_individuals\n"
    n_mutation=len(recommend_algorithm)-1
    
    mutation_instruction="Independently make improvements over these Functions that will increase their practical performance (not on the code efficiency, readability and parallel processing level). You are encouraged to be creative to incorporate novel ideas."

    if n_mutation==1:
        output_instruction="Provide runnable code that has implemented all your ideas (If any part requires choice, use choices from random). Do not leave any placeholder for me, all the functionality should be actually implemented. Your response format\n<code>(must include this sign)\nYour code\n</code>(must include this sign)\nAlso, you don’t need to add any other words.\n"

    else:
        output_instruction="Provide runnable code that has implemented all your ideas (If any part requires choice, use choices from random). Do not leave any placeholder for me, all the functionality should be actually implemented. Your response format\n<code>(must include this sign)\nYour code 1\n</code>(must include this sign)\n...\n<code>(must include this sign)\nYour code {}\n</code>(must include this sign)\nAlso, you don’t need to add any other words and name each code 'GenerateSample'.\n".format(n_mutation)

    meta_prompt=mutation_instruction+output_instruction

    
    code_wrong_flag=True
    wrong_flag=True
    exec_wrong_flag=True
    find_no_code_flag=True

    LLM_wrong_flag=True
    re_prompt_num=0
    while (re_prompt_num<5)&((code_wrong_flag==True) | (wrong_flag==True)| (exec_wrong_flag==True)| (find_no_code_flag==True)|  (LLM_wrong_flag==True)):
        code_wrong_flag=False
        wrong_flag=False
        exec_wrong_flag=False
        find_no_code_flag=False
        LLM_wrong_flag=False
        re_prompt_num=re_prompt_num+1
        
        tnp=""
        for algorithm_index in range(n_mutation):
            tnp=tnp+"Function {}:\n".format(algorithm_index+1)
            tnp=tnp+recommend_algorithm[algorithm_index+1]['alg_code']+"\n"

        algorithms_prompt=tnp


        client = OpenAI(

                api_key=open("./Key.txt", "r").read().strip("\n"),

                base_url="https://api.aiproxy.io/v1"
            )
        LLM_wrong_flag=False


        in_context_prompt=function_description+algorithms_prompt+meta_prompt

        try:
            completion = client.chat.completions.create(
              model=gpt_model, 
              temperature=0.7,
              messages=[{"role": "user", "content": in_context_prompt}]
            )
        except:
            LLM_wrong_flag=True

        if LLM_wrong_flag !=True:
            find_no_code_flag=False


            pattern = re.compile(r'<code>\n(.*?)\n</code>',re.DOTALL)
            match = pattern.findall(completion.choices[0].message.content)
            if match:
                    novel_alg_code_list=match
            else:
                    find_no_code_flag=True



            if (find_no_code_flag!=True):
              working_algorithms_index=[]
              problematical_codes=0
              for algorithm_index in range(len(novel_alg_code_list)):
                novel_alg_code=novel_alg_code_list[algorithm_index]
                added_indent='            '

                original_string=novel_alg_code
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


                exec_wrong_flag=False

                try:
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
                    exec(function_code, globals(), local_vars)
                    # Retrieve the defined function from local_vars
                    GenerateSample = local_vars['GenerateSample']
                except:
                    exec_wrong_flag=True
                if exec_wrong_flag!=True:
                    code_length=10
                    new_individuals_numbers=50
                    maximum_iteration=30
                    current_iteration=np.random.randint(2,maximum_iteration+1)
                    hyperparameters={}
                    history_populations={}
                    fitness_scores={}
                    hyperparameters.update({'code_bound':10})
                    hyperparameters.update({'code_upperbound': 10})
                    BF=1000000
                    code_wrong_flag=False
                    ## Generate individuals and their fitness scores
                    for i in range(current_iteration-1):
                        history_populations.update({'{}'.format(i+1):[]})
                        fitness_scores.update({'{}'.format(i+1):[]})
                        for j in range(np.random.randint(10,new_individuals_numbers+1)):
                            history_populations['{}'.format(i+1)].append(np.random.randint(1,hyperparameters['code_bound']+1,[code_length]))
                            fitness_scores['{}'.format(i+1)].append(float(np.abs(np.random.randn())))
                            if fitness_scores['{}'.format(i+1)][-1]<BF:
                                BF=fitness_scores['{}'.format(i+1)][-1]
                                best_individual=history_populations['{}'.format(i+1)][-1].copy()

                    try:
                        new_individuals=GenerateSample(history_populations, fitness_scores, best_individual, new_individuals_numbers, current_iteration, maximum_iteration, hyperparameters)
                    except:
                        code_wrong_flag=True
                    if code_wrong_flag !=True:
                        ## Check new_individuals
                        if code_wrong_flag==False:
                            wrong_flag=False
                            for i in range(len(new_individuals)):
                                tnp=new_individuals[i].copy()
                                tnp=np.array(tnp).reshape(-1)
                                if len(tnp)!=code_length:
                                    wrong_flag=True
                                    break


                if ((exec_wrong_flag==True)| (code_wrong_flag==True)| (wrong_flag==True))==True:
                                problematical_codes=problematical_codes+1
                                exec_wrong_flag=False
                                code_wrong_flag=False
                                wrong_flag=False
                else:
                                working_algorithms_index.append(algorithm_index)
            if len(working_algorithms_index)==0:
                wrong_flag=True
            else:
                exec_wrong_flag=False
                code_wrong_flag=False
                wrong_flag=False







    if ((code_wrong_flag==True) | (wrong_flag==True)| (exec_wrong_flag==True)| (find_no_code_flag==True)| (LLM_wrong_flag==True))==False:

        for i in range(len(working_algorithms_index)):
            novel_alg_code=novel_alg_code_list[working_algorithms_index[i]]
            recommend_algorithm[working_algorithms_index[i]+1]['alg_code']=novel_alg_code



    return recommend_algorithm




def Post_processing(recommend_algorithm,file_name,gpt_model):
    # file_path = "./my_algorithms_pool.pkl"
    file_path="./"+file_name+"/my_algorithms_pool.pkl"
# Read the dictionary from the binary file
    with open(file_path, 'rb') as pickle_file:
        populations_of_algorithms = pickle.load(pickle_file)
        

    # file_path = "./my_algorithms_searching.pkl"
    file_path="./"+file_name+"/my_algorithms_searching.pkl"
# Read the dictionary from the binary file
    with open(file_path, 'rb') as pickle_file:
        algorithms_search_process = pickle.load(pickle_file)


    n_generate=len(recommend_algorithm)-1

    meta_promot="Which function in the above is methodologically most similar to the new function? Just give me the function number with no other words.\n"

    cluster_num=len(populations_of_algorithms)
    clusters_size=np.arange(cluster_num)
    clusters_fit=[]
    clusters_best_fit=[]
    clusters_best_fit_algo_num=[]
    for i in range(cluster_num):
            clusters_size[i]=int(len(populations_of_algorithms['cluster{}'.format(i+1)])//3)
            tnp=[]
            for j in range(int(len(populations_of_algorithms['cluster{}'.format(i+1)])//3)):
                tnp.append(populations_of_algorithms['cluster{}'.format(i+1)]['fit_alg{}'.format(j+1)])
            clusters_fit.append(tnp.copy())
            clusters_best_fit.append(min(tnp))
            clusters_best_fit_algo_num.append(tnp.index(min(tnp))+1)

    tnp=""
    count=0
    for cluster_index in range(cluster_num):
                count=count+1
                tnp=tnp+"Function {}:\n".format(count)
                tnp=tnp+populations_of_algorithms['cluster{}'.format(count)]['code_alg{}'.format(clusters_best_fit_algo_num[count-1])]+"\n"

    algorithms_clusters=tnp+"New Function:\n"
    
    goto_cluster=[]
    goto_cluster_index=[]

    for algorithm_index in range(n_generate):
        
      find_no_code_flag=True
      LLM_wrong_flag=True
      re_prompt_num=0
      while (re_prompt_num<5)&((find_no_code_flag==True)|  (LLM_wrong_flag==True)):

          find_no_code_flag=False
          LLM_wrong_flag=False
          re_prompt_num=re_prompt_num+1
          


          algorithms_prompt=algorithms_clusters+recommend_algorithm[algorithm_index+1]['alg_code']+"\n"


          client = OpenAI(

                  api_key=open("./Key.txt", "r").read().strip("\n"),

                  base_url="https://api.aiproxy.io/v1"
              )
          LLM_wrong_flag=False


          in_context_prompt=algorithms_prompt+meta_promot


          try:
              completion = client.chat.completions.create(
                model=gpt_model, 
                temperature=0.7,
                messages=[{"role": "user", "content": in_context_prompt}]
              )
          except:
              LLM_wrong_flag=True

          if LLM_wrong_flag !=True:
              find_no_code_flag=False


              pattern = re.compile(r'\d+')
              match = pattern.search(completion.choices[0].message.content)

              if match:
                  first_number = int(match.group())  # This will contain the first number found

              else:
                  find_no_code_flag=True  # No number found
              if find_no_code_flag !=True:
                if ((first_number>=1)&(first_number<=cluster_num))==False:
                    find_no_code_flag=True



      if ((find_no_code_flag==True)| (LLM_wrong_flag==True))==False:
          goto_cluster.append(first_number)
          goto_cluster_index.append(clusters_size[goto_cluster[-1]-1]+1)
          clusters_size[goto_cluster[-1]-1]=clusters_size[goto_cluster[-1]-1]+1
      else:
          goto_cluster.append(np.random.randint(1, cluster_num+1))
          goto_cluster_index.append(clusters_size[goto_cluster[-1]-1]+1)
          clusters_size[goto_cluster[-1]-1]=clusters_size[goto_cluster[-1]-1]+1



    
    ## update the algorithms_search_process
    
    algorithms_search_process['Step{}'.format(len(algorithms_search_process))].update({'Gotocluster{}'.format(goto_cluster):goto_cluster_index})
    ### save the new population locally
    # file_path = "./my_algorithms_searching.pkl"
    file_path="./"+file_name+"/my_algorithms_searching.pkl"
    # Save the dictionary to a JSON file
    with open(file_path, 'wb') as pickle_file:
            pickle.dump(algorithms_search_process, pickle_file)

    ## update the populations_of_algorithms
    for algorithm_index in range(n_generate):
        populations_of_algorithms['cluster{}'.format(goto_cluster[algorithm_index])].update({'code_alg{}'.format(goto_cluster_index[algorithm_index]):recommend_algorithm[algorithm_index+1]['alg_code']})
        populations_of_algorithms['cluster{}'.format(goto_cluster[algorithm_index])].update({'descri_alg{}'.format(goto_cluster_index[algorithm_index]):recommend_algorithm[algorithm_index+1]['alg_descri']})
        populations_of_algorithms['cluster{}'.format(goto_cluster[algorithm_index])].update({'fit_alg{}'.format(goto_cluster_index[algorithm_index]):recommend_algorithm[algorithm_index+1]['alg_fitness']})


    ### save the new population locally
    # file_path = "./my_algorithms_pool.pkl"
    file_path="./"+file_name+"/my_algorithms_pool.pkl"
    # Save the dictionary to a JSON file
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(populations_of_algorithms, pickle_file)

    return goto_cluster, goto_cluster_index



def generate_novel_cluster(file_name,gpt_model):


    # file_path = "./my_algorithms_pool.pkl"
    file_path="./"+file_name+"/my_algorithms_pool.pkl"
# Read the dictionary from the binary file
    with open(file_path, 'rb') as pickle_file:
        populations_of_algorithms = pickle.load(pickle_file)

    function_description="Function description:\ndef GenerateSample(history_populations,fitness_scores,best_individual,new_individuals_numbers,current_iteration,maximum_iteration,hyperparameters):\n     ## GenerateSample: Function takes in integer vectors and output integer vectors.\n     ## Input: \n     ## history_populations: Dictionary. Keys are integer strings from '1' to some larger value. Each key contains a list of integer numpy vectors.\n     ## fitness_scores: Dictionary. Keys are same as history_populations. Each key contains a list of floats, the lower the better.\n     ## best_individual: Numpy integer vector.\n     ## new_individuals_numbers: Integer.\n     ## current_iteration: Integer, 1 larger than len of history_populations.\n     ## maximum_iteration: Integer.\n     ## hyperparameters: Dictionary. Keys are strings contain the constant used for computation. Default values should be provided using .get().\n     ## Output: \n     ## new_individuals: List, len new_individuals_numbers, contains integer numpy vectors. Each vector's len is the same as the len of the vectors in history_populations. Furthermore, The elements are within range [1,hyperparameters['code_upperbound']].\n     return new_individuals\n"

    cluster_instruction="Give me a novel ‘GenerateSample’ that is methodologically different from the above functions. You are encouraged to be creative to incorporate novel ideas but do not simply stack methods together."

    output_instruction=" Provide runnable code that has implemented all your ideas (If any part requires choice, use choices from random). Do not leave any placeholder for me, all the functionality should be actually implemented. Your response format\n<code>(must include this sign)\nYour code\n</code>(must include this sign)\nAlso, you don’t need to add any other words.\n"


    meta_prompt=cluster_instruction+output_instruction
    

    code_wrong_flag=True
    wrong_flag=True
    exec_wrong_flag=True
    find_no_code_flag=True

    LLM_wrong_flag=True
    re_prompt_num=0
    while (re_prompt_num<5)&((code_wrong_flag==True) | (wrong_flag==True)| (exec_wrong_flag==True)| (find_no_code_flag==True)| (LLM_wrong_flag==True)):
        code_wrong_flag=False
        wrong_flag=False
        exec_wrong_flag=False
        find_no_code_flag=False

        LLM_wrong_flag=False
        re_prompt_num=re_prompt_num+1
        cluster_num=len(populations_of_algorithms)
        clusters_size=np.arange(cluster_num)
        clusters_fit=[]
        clusters_best_fit=[]
        clusters_best_fit_algo_num=[]
        for i in range(cluster_num):
            clusters_size[i]=int(len(populations_of_algorithms['cluster{}'.format(i+1)])//3)
            tnp=[]
            for j in range(int(len(populations_of_algorithms['cluster{}'.format(i+1)])//3)):
                tnp.append(populations_of_algorithms['cluster{}'.format(i+1)]['fit_alg{}'.format(j+1)])
            clusters_fit.append(tnp.copy())
            clusters_best_fit.append(min(tnp))
            clusters_best_fit_algo_num.append(tnp.index(min(tnp))+1)

        tnp=""
        count=0
        for cluster_index in range(cluster_num):
                count=count+1
                tnp=tnp+"Function {}:\n".format(count)
                tnp=tnp+populations_of_algorithms['cluster{}'.format(count)]['code_alg{}'.format(clusters_best_fit_algo_num[count-1])]+"\n"

        algorithms_prompt=tnp

        client = OpenAI(

            api_key=open("./Key.txt", "r").read().strip("\n"),

            base_url="https://api.aiproxy.io/v1"
        )

        LLM_wrong_flag=False


        in_context_prompt=function_description+algorithms_prompt+meta_prompt

        try:
            completion = client.chat.completions.create(
              model=gpt_model, 
              temperature=0.7,
              messages=[{"role": "user", "content": in_context_prompt}]
            )
        except:
            LLM_wrong_flag=True


        if LLM_wrong_flag !=True:

            find_no_code_flag=False
            pattern = re.compile(r'<code>\n(.*?)\n</code>',re.DOTALL)
            match = pattern.search(completion.choices[0].message.content)
            if match:
                    novel_alg_code=match.group(1)
            else:
                matches = re.findall(r'```python(.*?)```', completion.choices[0].message.content, re.DOTALL)
                if matches==[]:
                            find_no_code_flag=True
                else:
                            novel_alg_code=matches[0]


            if (find_no_code_flag!=True):
                ### add indent befer running code of string
                added_indent='            '

                original_string=novel_alg_code
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


                exec_wrong_flag=False

                try:
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
                    exec(function_code, globals(), local_vars)
                    # Retrieve the defined function from local_vars
                    GenerateSample = local_vars['GenerateSample']
                except:
                    exec_wrong_flag=True
                if exec_wrong_flag!=True:
                    code_length=10
                    new_individuals_numbers=50
                    maximum_iteration=30
                    current_iteration=np.random.randint(2,maximum_iteration+1)
                    hyperparameters={}
                    history_populations={}
                    fitness_scores={}
                    hyperparameters.update({'code_bound':10})
                    hyperparameters.update({'code_upperbound': 10})
                    BF=1000000
                    code_wrong_flag=False
                    ## Generate individuals and their fitness scores
                    for i in range(current_iteration-1):
                        history_populations.update({'{}'.format(i+1):[]})
                        fitness_scores.update({'{}'.format(i+1):[]})
                        for j in range(np.random.randint(10,new_individuals_numbers+1)):
                            history_populations['{}'.format(i+1)].append(np.random.randint(1,hyperparameters['code_bound']+1,[code_length]))
                            fitness_scores['{}'.format(i+1)].append(float(np.abs(np.random.randn())))
                            if fitness_scores['{}'.format(i+1)][-1]<BF:
                                BF=fitness_scores['{}'.format(i+1)][-1]
                                best_individual=history_populations['{}'.format(i+1)][-1].copy()

                    try:
                        new_individuals=GenerateSample(history_populations, fitness_scores, best_individual, new_individuals_numbers, current_iteration, maximum_iteration, hyperparameters)
                    except:
                        code_wrong_flag=True
                    if code_wrong_flag !=True:
                        ## Check new_individuals
                        if code_wrong_flag==False:
                            wrong_flag=False
                            for i in range(len(new_individuals)):
                                tnp=new_individuals[i].copy()
                                tnp=np.array(tnp).reshape(-1)
                                if len(tnp)!=code_length:
                                    wrong_flag=True
                                    break
                        ## need to make sure all the 'wrong flag' are false 'code_wrong_flag', 'wrong_flag','exec_wrong_flag','find_no_code_flag','find_no_code_descri_flag','LLM_wrong_flag'



    if ((code_wrong_flag==True) | (wrong_flag==True)| (exec_wrong_flag==True)| (find_no_code_flag==True)|(LLM_wrong_flag==True))==False:

        populations_of_algorithms.update({'cluster{}'.format(cluster_num+1):{}})

        populations_of_algorithms['cluster{}'.format(cluster_num+1)].update({'code_alg{}'.format(1):novel_alg_code})
        populations_of_algorithms['cluster{}'.format(cluster_num+1)].update({'descri_alg{}'.format(1):''})
        populations_of_algorithms['cluster{}'.format(cluster_num+1)].update({'fit_alg{}'.format(1):10000000})

        ### save the new population locally
        # file_path = "./my_algorithms_pool.pkl"
        file_path="./"+file_name+"/my_algorithms_pool.pkl"
        # Save the dictionary to a JSON file
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(populations_of_algorithms, pickle_file)
















