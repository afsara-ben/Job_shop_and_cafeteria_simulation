"""

Job shop model
See assignment specs for detailed description

"""

import heapq
import random
import matplotlib.pyplot as plt
import linecache as lc
import numpy as np


Q_LIMIT = 10
IDLE = 0
BUSY = 1

MAX_NUM_STATIONS = 5
MAX_NUM_JOB_TYPES = 3

inputFile = "input.txt"



# Parameters
class Params:
    def __init__(self, lambd, mu, k):
        self.lambd = lambd  # interarrival rate
        self.mean_service = mu # mean_service[i][j] is mean service time of task j for job i
        self.k = k
    # Note lambd and mu are not mean value, they are rates i.e. (1/mean)


# States and statistical counters
class States:
    def __init__(self, num_stations, num_job_types, prob_dist_job_type, num_machines, num_tasks, route):
        # job shop variables
        self.num_stations = num_stations
        self.num_job_types = num_job_types
        self.prob_dist_job_type = prob_dist_job_type
        self.num_machines = num_machines #num_machines[MAX_NUM_STATIONS + 1]
        self.num_tasks = num_tasks #num_tasks[MAX_NUM_JOB_TYPES + 1]
        self.route = route #route[MAX_NUM_JOB_TYPES + 1][MAX_NUM_STATIONS + 1]
        self.num_machines_busy = [0] * (MAX_NUM_STATIONS) #number of machines currently busy in station j


        # State variables
        self.server_status = [] #list state of k servers
        self.station_delay = [0] * (MAX_NUM_STATIONS)
        self.job_delay = [0] * (MAX_NUM_JOB_TYPES)
        self.no_of_delay =0.0
        self.num_custs_delayed = 0.0
        self.queue = []                                     #stores the arrival times of each job
        # self.queue.append([]) #2d array row = server #, queue[i][] = customers in server row[i] 


        # intermediate variables
        self.arrival_event_count = [0] * (MAX_NUM_JOB_TYPES )
        self.departure_event_count = 0
        self.time_of_last_event = 0.0
        self.Total_time_served = 0.0
        self.Area_under_queue = [0] * MAX_NUM_STATIONS
        self.curr_amount_of_jobs = [0] * (MAX_NUM_JOB_TYPES)
        self.Area_under_num_jobs = 0.0
       
        # Statistics
        self.util = 0.0
        self.avgQdelay = [0] * MAX_NUM_STATIONS
        self.avgQlength = [0] * MAX_NUM_STATIONS
        self.avgJobDelay = [0] * MAX_NUM_JOB_TYPES
        self.served = [0] * (MAX_NUM_STATIONS )
        self.avgJobCount = 0.0
        self.overallDelay = 0.0


    def print_init(self,sim):
        print("lambda : ", sim.params.lambd)
        print("Number of jobs = " , sim.states.num_job_types)
        print("Probability of jobs ", sim.states.prob_dist_job_type)
        print("Number of stations = " , sim.states.num_stations)
        print("Number of machines in each station ", sim.states.num_machines)
        print("Number of tasks for each job ", sim.states.num_tasks)
        for i in range(int(sim.states.num_job_types)):
            print()
            print("Routing of job " + str(i+1))
            print(sim.states.route[i+1])
            print("Job " + str(i+1) + " mean service time for each station")
            print(sim.params.mean_service[i+1])



    def set_server_and_queue_status(self,sim):
        print("Initializing ", sim.params.k, " servers and ", sim.params.k, " queues")
        for k in range(sim.params.k):
            sim.states.server_status.append(IDLE) #server[i] = IDLE initially
            sim.states.queue.append([]) #list of list

        self.see_server_status(sim)



    def see_server_status(self,sim):
        for i in range(sim.params.k):
            print('Station ', i+1, ' - ', len(sim.states.queue[i]))


    def print_queue_status(self, sim):
        for i in range(sim.params.k):
            # print( 'queue ',i, ' - ', sim.states.queue_status[i])
            print( 'queue ',i, ' - ', len(sim.states.queue[i]))


    def get_avg_len_of_queues(self,sim):
        k = sim.params.k
        len_ = 0

        for i in range(k):
            len_ += len(sim.states.queue[i])

        return len_/k


    def update(self, sim, event):
        # Complete this function

        
        # print("\nin update() of States\n")
        print('\nUpdated stats\n')
        
        
        time_since_last_event = event.eventTime - sim.states.time_of_last_event # time_since_last_event = duita event er modhekar time
                                                                                #event.eventTime = oi event er time
        
        sim.states.time_of_last_event = event.eventTime

        station = 0
        if event.eventType != "START":
            station = int(sim.states.route[event.job_type][event.task-1])

        # no_of_busy_servers = self.count_busy_servers(sim)

        
        # mult = (no_of_busy_servers/sim.params.k)
        # sim.states.Total_time_served +=  mult * time_since_last_event

        # avg_lenQ = self.get_avg_len_of_queues(sim)
        # sim.states.Area_under_queue += avg_lenQ * time_since_last_event

        sim.states.Area_under_queue[station-1] += len(sim.states.queue[station - 1]) * time_since_last_event

        total_curr_jobs = 0.0
        for i in range(MAX_NUM_JOB_TYPES):
            total_curr_jobs +=  sim.states.curr_amount_of_jobs[i]

        sim.states.Area_under_num_jobs += total_curr_jobs * time_since_last_event

        
        
        print("Current jobs in system - ", sim.states.curr_amount_of_jobs)
        print('Total_time_served - ', sim.states.Total_time_served)
        print('Area_under_queue - ', sim.states.Area_under_queue)
        print('Area_under_num_jobs - ', sim.states.Area_under_num_jobs)
        print('Total tasks served - ', sim.states.served)
       
        print()
     

    def finish(self, sim):
        

        # self.avgQlength = self.Area_under_queue/sim.simclock

        for i in range(MAX_NUM_STATIONS):
            self.avgQlength[i] = self.Area_under_queue[i]/sim.simclock

        
        for i in range(MAX_NUM_STATIONS):
            if sim.states.served[i]:
                self.avgQdelay[i] = self.station_delay[i]/sim.states.served[i]


        for i in range(MAX_NUM_JOB_TYPES):
            if sim.states.arrival_event_count[i]:
                self.avgJobDelay[i] = sim.states.job_delay[i]/sim.states.arrival_event_count[i]

        # calculate avg jobs in system
        self.avgJobCount = self.Area_under_num_jobs/sim.simclock

        # calculate overall delay
        for i in range(MAX_NUM_JOB_TYPES):
            self.overallDelay += sim.states.prob_dist_job_type[i] * self.avgJobDelay[i]

        

       
       

    def printResults(self, sim):
        print()


        
        # print ('All arrival times  - ', sim.states.queue)
        # DO NOT CHANGE THESE LINES
        print('MMk Results: lambda = ', sim.params.lambd, "mu = ", sim.params.mean_service)
        print('Job probabilities :', sim.states.prob_dist_job_type)
        # print('MMk Total delay: %d' % (sim.states.total_delays))
        
        print('\nArrival_event_count- ', sim.states.arrival_event_count)
        print( 'Departure_event_count- ', sim.states.departure_event_count)
        print('Total tasks served:',  sim.states.served)
        self.avgQdelay = [ '%.5f' % elem for elem in self.avgQdelay ]
        self.avgJobDelay = [ '%.5f' % elem for elem in self.avgJobDelay ]
        self.avgQlength = [ '%.5f' % elem for elem in self.avgQlength ]
        print('Average queue length: ' , self.avgQlength)
        print('Average task delay in queue: ', self.avgQdelay)
        print('Average job delay ', self.avgJobDelay)

        print('Average number of jobs', self.avgJobCount)
        print('Overall job delay', self.overallDelay)
   


    def getResults(self, sim):
        return (self.avgQlength, self.avgQdelay, self.avgJobDelay, self.avgJobCount, self.overallDelay)



class Event:
    def __init__(self, sim):
        self.eventType = None
        self.sim = sim
        self.eventTime = None
        self.curr_task = None
        self.curr_job_type = None

    def process(self, sim):
        raise Exception('Unimplemented process method for the event!')

    def __repr__(self):
        return self.eventType


    #this class decorater needed because of this comparison
    def __lt__(self, other):
        return self.eventTime < other.eventTime


class StartEvent(Event):
    def __init__(self, eventTime, sim):
        self.eventTime = eventTime
        self.eventType = 'START'
        self.sim = sim

    def process(self, sim):
        print ("in StartEvent's process()")

        
        # sim.states.see_server_status(sim)

        # determine start job type
        first_job_type = determine_job_type(sim)
        first_task = 1
        sim.scheduleEvent(ArrivalEvent(random.expovariate(sim.params.lambd), first_job_type, first_task, sim, new_job=1)) #initially start at 0th server
        # sim.scheduleEvent(ArrivalEvent(sim.simclock, first_job_type, first_task, sim, new_job=1)) #initially start at 0th server
        
        sim.scheduleEvent(ExitEvent(int(8), sim))
        

class ExitEvent(Event):
    def __init__(self, eventTime, sim):
        self.eventTime = eventTime
        self.eventType = 'EXIT'
        self.sim = sim

    def process(self, sim):
        None


def determine_job_type(sim):
    # determine the job type from given probability sampling
    job_type = np.random.choice(np.arange(1, 4), p=sim.states.prob_dist_job_type)
    print("job type : ", job_type)
    return job_type

#  /* Function to serve as both an arrival event of a job to the system, 
# as well as the non-event of a job's
# arriving to a subsequent station along its route. */

class ArrivalEvent(Event):
    def __init__(self,eventTime, curr_job_type, curr_task, sim, new_job):
        
        self.eventType = 'ARRIVAL'
        self.eventTime = eventTime
        self.sim = sim
        self.new_job = new_job

        self.job_type = curr_job_type
        self.task = curr_task

        # sim.states.curr_amount_of_jobs[self.job_type - 1] +=1

        if new_job == 1:
            print("\nInitializing new ArrivalEvent at %lf " % self.eventTime, "for job type ", self.job_type, " and task ", self.task)
        else:
            print("\nInitializing ArrivalEvent at %lf " % self.eventTime, "for job type ", self.job_type, " and task ", self.task)



    def process(self, sim):
        print("in process() of ArrivalEvent")

        # /* If this is a new arrival to the system, generate the time of the next
        # arrival and determine the job type and task number of the arriving
        # job. */

        if(self.new_job == 1):
            
            sim.states.curr_amount_of_jobs[self.job_type - 1] +=1
            timeOfNextArrival = sim.simclock + random.expovariate(sim.params.lambd)
            # print(' timeOfNextArrival ', timeOfNextArrival)

            # determine the job type
            next_job_type = determine_job_type(sim)

            # set task = 1
            next_task = 1
            new_job = 1

            
        
            # Schedule next new (n+1) arrival 
            sim.scheduleEvent(ArrivalEvent(timeOfNextArrival, next_job_type, next_task, sim, new_job)) 

            #increase arrival count of this job type
            sim.states.arrival_event_count[self.job_type-1] +=1
            # print("Total Arrival events of job type", self.job_type, "=", sim.states.arrival_event_count[self.job_type-1] )

            
            

        # Determine the station from the route matrix
        station = int(sim.states.route[self.job_type][self.task-1]) #as second index is 0 based 
        print("\nStation for current job", self.job_type, "- task ", self.task, ":", station)


        # Check to see whether all machines in this station are busy
        if(sim.states.num_machines_busy[station-1] == sim.states.num_machines[station - 1]):
            # /* All machines in this station are busy, so place the arriving job at
            # the end of the appropriate queue.

            print("\n~~~~~~~~~~~~~~~~~~~~~~~ ALL machines at station",station,"are now busy ~~~~~~~~~~~~~~~~~~~~`")
           
            print('After inserting customer at queue of station ', station, 'having q length ', len(sim.states.queue[station-1]))

            
            
            #put arrival time of this customer in its station's queue
            sim.states.queue[station-1].append(sim.simclock)
            # sim.states.queue_status[station-1] = len(sim.states.queue[station-1])


            sim.states.see_server_status(sim)

        else:
            # A machine in this station is idle, so start service on the arriving job (which has a delay of zero)
            
            # make a machine in this station busy
            print("\nBefore: Number of busy machines", sim.states.num_machines_busy)
            
            sim.states.num_machines_busy[station-1] +=1
            print("\nAfter: Number of busy machines", sim.states.num_machines_busy)

            sim.states.served[station-1] +=1
            
            # Schedule a departure (service completion)
            print("\nScheduling a departure event for current job ", self.job_type, "and task", self.task)
            sim.states.departure_event_count += 1 
            # self.task +=1 ????
            departure_time = sim.simclock + erlang_2(sim.params.mean_service[self.job_type][self.task-1])
            sim.scheduleEvent(DepartureEvent(departure_time, sim, curr_job_type=self.job_type, curr_task=self.task))
           


# Event function for departure of a job from a particular station
class DepartureEvent(Event):
    def __init__(self,eventTime, sim, curr_job_type, curr_task):
        print("Initializing DepartureEvent")
        self.eventType = 'DEPARTURE'
        self.eventTime = eventTime
        self.sim = sim
        self.job_type = curr_job_type
        self.task = curr_task
        


    def process(self, sim):
        print("in process() of DepartureEvent")

        # determine the station from which the job is departing
        station = int(sim.states.route[self.job_type][self.task-1])

        
        # sim.states.print_queue_status(sim)
        sim.states.see_server_status(sim)

        # Check to see whether the queue for this station is empty
        if(len(sim.states.queue[station-1]) == 0):

            print("The queue for station ", station, " is empty ")
            print("so decrease number of busy machines by 1")
            # The Queue for this station is empty, so make a machine in this staion idle
            sim.states.num_machines_busy[station-1] -= 1
            print("Number of busy machines", sim.states.num_machines_busy)
            # print('Q is empty, so make station', station ,' idle')
            # self.sim.states.server_status[station-1] = IDLE


        else:
            # The queue is nonempty, so start service on first job in queue. 
            print("The queue is nonempty, so start service on first job in queue ")
            sim.states.served[station-1] += 1
            print("Job served at station", sim.states.served[station-1])
            # Increment the number of customers delayed, and schedule departure. 
            # self.sim.states.num_custs_delayed +=1;

            # compute delay for this job and gather statistics
            sim.states.no_of_delay +=1
            # delay = sim.simclock - sim.states.queue[0][0] #curr time - time of arrival of this customer who is now being served
            delay_ = sim.simclock - sim.states.queue[station-1][0]
            sim.states.station_delay[station-1] += delay_
            sim.states.job_delay[self.job_type-1] += delay_

            # schedule a departure event for this job
            print("Scheduling a departure event for job ",self.job_type ,"task ", self.task)
            time_departure_next_event = sim.simclock + erlang_2(sim.params.mean_service[self.job_type][self.task-1])
            sim.states.departure_event_count +=1
            sim.scheduleEvent(DepartureEvent(time_departure_next_event, sim, self.job_type, self.task))

            # remove the first job from the queue
            sim.states.queue[station-1] = sim.states.queue[station-1][1:]
            
        # if job is in final station, return
        # else add 1 to task for the departing job -> invoke arrive with new_job = 2
    
        if(self.task == int(sim.states.num_tasks[self.job_type-1])): #means job is in final station
            # decrement that job from current job count
            sim.states.curr_amount_of_jobs[self.job_type - 1] -=1
            print("Job is now in final station")
            return
        else:
        # /* If the current departing job has one or more tasks yet to be done, send
        # the job to the next station on its route. */    
            print("Job is not in final station so move job to next station on its route")
            self.task += 1;
            sim.scheduleEvent(ArrivalEvent(sim.simclock, self.job_type, self.task, sim, new_job = 2))

       


class Simulator:
    def __init__(self, seed):
        print("in init of Simulator")
        self.eventQ = []
        self.simclock = 0
        self.seed = seed
        self.params = None
        self.states = None

    def initialize(self):
        print("Initializing Simulator")
        self.simclock = 0
        self.states.set_server_and_queue_status(self)  #-- changed here

        self.scheduleEvent(StartEvent(0, self))

    def configure(self, params, states):
        print("Configuring Simulator")
        self.params = params
        self.states = states
        self.states.print_init(self)

    def now(self):
        print("in now of Simulator")
        return self.simclock

    def scheduleEvent(self, event):
        # print("in scheduleEvent of Simulator")
        heapq.heappush(self.eventQ, (event.eventTime, event))

    def run(self):
        print("Running Simulator")
        random.seed(self.seed)
        self.initialize()

        
        while len(self.eventQ) > 0:
            # print('len of eventQ ' ,len(self.eventQ))
            time, event = heapq.heappop(self.eventQ)
            # print('popping ', event, ' at ', time)

            if event.eventType == 'EXIT':
                print ("\nEXITING")
                break

            if self.states != None:
                self.states.update(self, event) #sim instance er states attr contains a States() instance which calls the update of States class

            print('\n-------------------------------------------------------------------------------')
            print('              At %lf' % event.eventTime, 'Event', event, 'k = ', self.params.k,'\n')
            
            self.simclock = event.eventTime
            event.process(self)

        self.states.finish(self)

    def printResults(self):
        self.states.printResults(self)

    def getResults(self):
        print("in getResults of Simulator")
        return self.states.getResults(self)

def erlang_2(mean):
    return random.expovariate(float(mean)/2)+random.expovariate(float(mean)/2)

def getline_(lineNumber):
    line = lc.getline(inputFile, lineNumber)
    return line

def read_input_from_file():

    #get number of stations 
    line = getline_(1)
    num_stations = int(line.split()[0])

    # get number of machines in each station
    num_machines = []
    line = getline_(2)
    for var in line.split():
        # upon encountering # stop reading that line
        if "#" in var: 
            break 
        num_machines.append(int(var))
    


    # get number of jobs
    line = getline_(4)
    num_job_types = int(line.split()[0])

    # get number of tasks for each job
    num_tasks = []
    line = getline_(6)
    for var in line.split():
        # upon encountering # stop reading that line
        if "#" in var: 
            break 
        num_tasks.append(int(var))
    
    #get interarrival rate for jobs
    line = getline_(3)
    lambd = 1/float(line.split()[0])

    #get job probabilities
    line = getline_(5)
    prob_dist_job_type = []
    for var in line.split():
        if "#" in var: 
            break 
        prob_dist_job_type.append(float(var))

    
    
    curr_line_number = 6
    route = {}
    mean_service = {}

    job_no = 1 

    for j in range(1, int(num_job_types)*2 ,2):
        line = getline_(curr_line_number + j)
        route_lst = []
        service_time_lst = []

        # get station routing of jobs
        for var in line.split():
            if "#" in var: 
                break 
            route_lst.append(int(var))
        route[job_no] = route_lst

        # get mean service times for num_job_types, 2d array
        line = getline_(curr_line_number + (j+1) )
        for var in line.split():
            if "#" in var: 
                break 
            service_time_lst.append(float(var))
        mean_service[job_no] = service_time_lst

        job_no +=1

    return num_stations, num_job_types, prob_dist_job_type, num_machines, num_tasks, route, lambd, mean_service


def job_shop_model():

    # read input from file
    num_stations, num_job_types, prob_dist_job_type, num_machines, num_tasks, route, lambd, mean_service = read_input_from_file()

    
   
    print("starting job shop model")
    seed = 1
    k=5
    
    avgQlength = [0] * MAX_NUM_STATIONS 
    avgQdelay = [0] * MAX_NUM_STATIONS
    avgJobDelay =[0]*MAX_NUM_JOB_TYPES
    avgJobCount = 0.0
    overallAvgDelay = 0.0
    
    itr = 30
    for i in range(itr):
        sim = Simulator(seed)
        states = States(num_stations, num_job_types, prob_dist_job_type, num_machines, num_tasks, route)
        sim.configure(Params(lambd, mean_service, k), states)
        sim.run()
        sim.printResults()
        Qlength, Qdelay, JobDelay, avgJobCount_, overallDelay_ = sim.getResults() #get the avg delay and   Qlength of for one iteration
                                                    # str keno return kore getResults() ???
        avgQlength = [x + float(y) for x, y in zip(avgQlength, Qlength)]
        avgQdelay = [x + float(y) for x, y in zip(avgQdelay, Qdelay)] 
        avgJobDelay = [x + float(y) for x, y in zip(avgJobDelay, JobDelay)]
        avgJobCount += avgJobCount_
        overallAvgDelay += overallDelay_
        # print('Iteration ', i , 'ended')

    avgQdelay = [ round( float(elem)/itr, 5) for elem in avgQdelay ]
    avgJobDelay = [ round( float(elem)/itr, 5) for elem in avgJobDelay ]
    avgQlength = [ round( float(elem)/itr, 5) for elem in avgQlength ]
    avgJobCount /= itr
    overallAvgDelay /=itr

    
   
    print('\n\nFinal Average queue length: ' , avgQlength)
    print('Final Average task delay in queue: ', avgQdelay)
    print('Final Average job delay ', avgJobDelay)
    print('Average number of jobs: %lf' % avgJobCount)
    print('Overall avg delay: %lf' %overallAvgDelay)
   




def main():
    job_shop_model();

if __name__ == "__main__":
    main()
