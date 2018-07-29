def _load_file(self, filepath, pg_val):
            proc = psutil.Process(os.getpid())
            perf={}
            speed = []
            mem = {}
            mem_usage = []
            mem_usage2 = []
            chunk_size = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000]
            for chunk in chunk_size:
                perf[chunk] = []
                mem[chunk] = []
                for run in range(0,2):
                    self.data = pd.DataFrame()
                    start_time = time.time()
                    i = 0
                    tmp = []
                    for df in pd.read_json(filepath, lines=True, chunksize=chunk): 
                        tmp.append(df)   
                        pg_val.set(i)
                        mem[chunk].append(proc.memory_info()[0]/(1024*1024))
                        i +=1
                    
                    self.data = pd.concat(tmp, axis=0)
          
                    self.done_loading = True  
                    print("Loading done")
                    mem_end = proc.memory_info()[0]/(1024*1024)
                    perf[chunk].append((time.time() - start_time, mem_end))
                    print(f"{time.time() - start_time}, {mem_end} MB")
            
            #print(perf)
            for chunk in chunk_size:    
                speed.append((chunk, mean([x[0] for x in perf[chunk]])))
            
            for chunk in chunk_size:    
                mem_usage.append((chunk, mean([x[1] for x in perf[chunk]])))
            
            for chunk in chunk_size:    
                mem_usage2.append((chunk, mean(mem[chunk])))
            
            
            print(speed)
            print(min(speed, key=lambda x:x[1]))
            
            print(mem_usage)
            print(min(mem_usage, key=lambda x:x[1]))
            
            print(mem_usage2)
            print(min(mem_usage2, key=lambda x:x[1]))