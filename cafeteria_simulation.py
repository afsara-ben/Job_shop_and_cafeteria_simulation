"""
Simulation of a cafeteria with hotfood, sandwich, drinks and cashier counter
the assignment specs contain the detailed description

"""

import heapq
import random
import matplotlib.pyplot as plt
import linecache as lc
import numpy as np



MAX_NUM_GROUPS = 4      #max 4 types of groups (1/2/3/4)
MAX_NUM_COUNTER = 4     #max 4 counters (hotfood/sandwich/drinks/cashier)
MAX_NUM_CUST_TYPES = 3  #max 3 types of customers (hotfood/sandwich/drinks)

#define
HOTDOG_COUNTER = 0
SANDWICH_COUNTER = 1
DRINK_COUNTER = 2
CASHIER_COUNTER = 3

inputFile = "input.txt"


# Parameters
class Params:
    def __init__(self, lambd, st, act, k):
        self.lambd = lambd  # interarrival rate
        self.st = st        #service time - a list of tuples
        self.act =  act     #accumulated cashier time - a list of tuples
        self.k = k          #number of employees at each counter - a list

    # Note lambd and mu are not mean value, they are rates i.e. (1/mean)

    def print__init__(self, sim):
        print("Lambda : ", sim.params.lambd)
        print("ST : ", sim.params.st)
        print("ACT : ", sim.params.act)
        print("Employees : ", sim.params.k)


# States and statistical counters
class States:
    def __init__(self, route, prob_dist_of_groups, prob_dist_cust_type, cashierCount):
        
        self.prob_dist_of_groups = prob_dist_of_groups          #probability distribution of each group (1/2/3/4)
        self.prob_dist_cust_type = prob_dist_cust_type          # probability dist of each customer type - (hotfood/sandwich/drinks)
        self.route = route                                      #route[customer_type][counter_task] - dict to keep track of which counter this customer should go now - 1 indexed
                        

        # State variables
        self.counter_status = []                                # list state of counters - size 4
        self.counter_delay = [0] * (MAX_NUM_COUNTER)            # delay at each counter
        self.customer_type_delay = [0] * (MAX_NUM_CUST_TYPES)   # delay for each customer type
        self.MAX_counter_delay = [0] * MAX_NUM_COUNTER          # max delay at each counter
        self.MAX_cust_type_delay = [0] * MAX_NUM_CUST_TYPES     # max delay for each customer type
        self.next_expected_group_id = 0                         # next expected id for arriving group - kind of like a token number
        self.queue = []                                         # stores the arrival times of each incoming group -  list of list of lists - queue [counterNo][queueNo] = [list of arrival time]
        self.total_custs_in_system = 0.0                        # total customers in system arrived so far - including those who departed from cashier
        self.curr_cust_of_each_type = [0] * MAX_NUM_CUST_TYPES  # number of customers in the system at this given instance of time

       

        #  intermediate variables
        self.arrival_event_count = [0] * MAX_NUM_GROUPS                     # arrival count of each group type  
        self.arrival_event_count_for_cust_type = [0] * MAX_NUM_CUST_TYPES   # arrival count for each customer type
        self.departure_event_count = 0                                      # departure even only occurs when the customer is in the cashier counter
        self.time_of_last_event = 0.0                                       # used to calculate time since last event
        self.avg_customer_in_system = 0.0                                   # area under customer - used to calculate avg customer in system after dividing by total time

       
        # Statistics
        self.overall_avg_delay = 0.0                                        # overall average delay for all customers
        self.avgQdelay_cust_type = [0] * MAX_NUM_CUST_TYPES                 # avg queue delay for each customer type
        self.avgQdelay_counter = [0] * MAX_NUM_COUNTER                      # avg queue delay for each counter type
        self.avgQlength = [0] * (MAX_NUM_COUNTER + cashierCount - 1)        # avg number of people in each queue at a time
        self.maxAvgQlength = [0] * (MAX_NUM_COUNTER + cashierCount - 1)     # max number of people count at each queue
        # self.avgGroupDelay = [0] * MAX_NUM_GROUPS
        self.served = [0] * (MAX_NUM_COUNTER )                              # count of service completion at each counter
        
       


    def print__init__(self,sim):
        print("lambda : ", sim.params.lambd)
        # print("Number of jobs = " , sim.states.num_job_types)
        print("Probability of each group ", sim.states.prob_dist_of_groups)
        print("Probability of each customer type ", sim.states.prob_dist_cust_type )
        
     
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ utility function of States class ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

    # check if the current counter has free employees or not
    # written such that if needed the food counters can also be made into multi queue counters. 
    def check_if_curr_counter_is_free(self, sim, curr_counter):

        print('\nChecking if counter', curr_counter, 'is free?')
        
        if curr_counter == DRINK_COUNTER :                                  #drinks counter - it is always free
            print('Drinks counter is  always free')
            return curr_counter, 0

        elif "IDLE" in self.counter_status[curr_counter]:                   # check if that counter is IDLE
            print('Counter ', curr_counter, ' is free')

            queueNo = self.counter_status[curr_counter].index("IDLE")       # find which queue of counter is free - for this simulation this is only helpful for cashier counter, food counters has only queue irrespective of employee count
            # print('returning counter', curr_counter, 'queueNo', queueNo ,'\n')
            
            return curr_counter, queueNo
            
        print ('Counter',curr_counter, 'is not free')
        return (-1,0)                                                       #means current counter is busy, default queue is 0

    def get_shortest_Q_of_cashier_counter(self,sim):
        lst = []
        for queueNo in range(sim.params.k[2]):
            lst.append(len(sim.states.queue[3][queueNo]))
        min_ = min(lst)
        
        return lst.index(min_)


    def set_counter_and_queue_status(self,sim):
        hotFood = sim.params.k[0]
        sandwich = sim.params.k[1]
        cashier = sim.params.k[2]

        print("Initializing ", hotFood, " hotFood employees, ", sandwich, " sandwich employees and ", cashier, " cashiers")
        
        for k in range(len(sim.params.k) + 1):
            sim.states.counter_status.append(["IDLE"])          # counter[i] = IDLE initially
            sim.states.queue.append([])                         
            sim.states.queue[k].append([])                      # list of list of lists

        
        
        # jotogula cashier ase tader jonne totogula server, IDLE initialize korlam
        # this initializing is twisted because k[] is of len 3 since drinks counter has no employee but i have made queue[] of len 4 -- but storing arrival time at drink counter is not necessary for this simulation
        for cashier in range(sim.params.k[2]-1):
            sim.states.counter_status[ CASHIER_COUNTER ].append("IDLE")
            sim.states.queue[ CASHIER_COUNTER ].append([])

        self.see_counter_status(sim)


    def see_counter_status(self,sim):
        # print('Showing queue matrix', sim.states.queue)

        for i in range(len(sim.params.k) + 1):
            print('Counter ', i, ' - ', sim.states.counter_status[i])

        self.print_queue_status(sim)
        self.print_cashier_queue_status(sim)


    def print_queue_status(self, sim):
        for i in range(len(sim.params.k)):
            print( 'queue len of counter',i, ' - ', len(sim.states.queue[i][0]))

    def print_cashier_queue_status(self, sim):
        for i in range(sim.params.k[2]):
            print( 'cashier queue len', i, ' - ', len(sim.states.queue[3][i]))

        print()

    
    def printCustomerName(self, cust_type):             # cust_type is 1-indexed
        if(cust_type == 1):
            print("Customer type: HOTFOOD CUSTOMER" )
        elif(cust_type == 2):
            print("Customer type: SANDWICH CUSTOMER" )
        elif(cust_type == 3):
            print("Customer type: DRINKS CUSTOMER" )
        else:
            print("this customer type does not exist")

    def printCounterName(self,counterNo):               # counterNo is 0-indexed
        if(counterNo == 0):
            print("Counter: HOTFOOD COUNTER\n")
        elif(counterNo == 1):
            print("Counter: SANDWICH COUNTER\n")
        elif(counterNo == 2):
            print("Counter: DRINKS COUNTER\n")
        elif(counterNo == 3):
            print("Counter: CASHIER COUNTER\n")
        else:
            print("this counter does not exist\n")


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ end of utility function of States class ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # update state variables and statistics
    def update(self, sim, event):
       
        print('\nUpdated stats\n')
        
        # Update the max number of customers in the system at any moment
        # Update the max length of the queues for each type of counters
 
        time_since_last_event = event.eventTime - sim.states.time_of_last_event # time_since_last_event = duita event er modhekar time
                                                                                # event.eventTime = oi event er time
        sim.states.time_of_last_event = event.eventTime

        counter = 0
        if event.eventType != "START":
            print('Customer arrival:', event.eventTime)
            self.printCustomerName(event.cust_type)
            print("Counter Task in route:",event.curr_counter_task)
            counter = sim.states.route[event.cust_type][event.curr_counter_task]
            self.printCounterName(counter)

            

        # sim.states.Area_under_queue[counter] += len(sim.states.queue[counter]) * time_since_last_event

        # total_curr_groups = 0.0
        # for i in range(MAX_NUM_GROUPS):
        #     total_curr_groups +=  sim.states.arrival_event_count[i]

        # sim.states.Area_under_group_num += total_curr_groups * time_since_last_event


        # for i in range(MAX_NUM_CUST_TYPES):
        #     curr_customers = self.curr_cust_of_each_type[i]
    
        
        # self.avg_customer_in_system += time_since_last_event * curr_customers

        self.avg_customer_in_system += time_since_last_event * self.total_custs_in_system
        
        

        sum_of_cashier_Q_len = 0
        if event.eventType != 'START':
            if counter != CASHIER_COUNTER:

                food_Q_len = len(sim.states.queue[counter][event.queueNo])
                self.maxAvgQlength[counter] = max(self.maxAvgQlength[counter], food_Q_len)
                self.avgQlength[counter] += food_Q_len * time_since_last_event

            else:

                # cashier q er avg q len ber korte chaile
                # for k in range(sim.params.k[2]):
                #     sum_of_cashier_Q_len += len(sim.states.queue[counter][k])

                # total_q_in_cashier_counter = sim.params.k[2]
                # self.maxAvgQlength[counter] = max(self.maxAvgQlength[counter], sum_of_cashier_Q_len)
                # self.avgQlength[counter] += (sum_of_cashier_Q_len / total_q_in_cashier_counter) * time_since_last_event
                    

                # all cashier q er alada alada q len ber korte chaile, or else uporer commented out code ta
                for k in range(sim.params.k[2]):
                    self.avgQlength[counter + k] += len(sim.states.queue[counter][k]) * time_since_last_event
                    self.maxAvgQlength[counter + k] = max(self.maxAvgQlength[counter + k], len(sim.states.queue[counter][k]))

        

        
        print("Groups in system so far - ", sim.states.arrival_event_count)
        print('Current customers of each type in system-', sim.states.curr_cust_of_each_type)
        print('Total current customers in system', self.total_custs_in_system)
        print('Average customer in system', self.avg_customer_in_system)
        print('Avg len of queue -',self.avgQlength )
        print('Total tasks served - ', sim.states.served)
        print('Departure event count -', sim.states.departure_event_count)
        print()
     


    def finish(self, sim):

        # print("in finish() of States class")

        # Avg queue delay for each counter : total delay / total served at that counter.
        for i in range( MAX_NUM_CUST_TYPES ):
            # print('cust type', i, ' delay %lf' % self.customer_type_delay[i], 'served', self.arrival_event_count_for_cust_type[i])
            if self.arrival_event_count_for_cust_type[i] == 0:
                self.avgQdelay_cust_type[i] = 0.0
            else:
                self.avgQdelay_cust_type[i] = self.customer_type_delay[i]/self.arrival_event_count_for_cust_type[i]
                self.avgQdelay_cust_type[i] /= 60 # convert to minutes
                self.MAX_cust_type_delay[i] /= 60 # convert maximum delay for each customer type in minutes

        


        # Avg queue delay for each type of customer: total delay / total arrival of this type of customer
        for i in range( MAX_NUM_COUNTER ):
            # print('counter', i, ' delay %lf' % self.counter_delay[i], 'served', self.served[i])
            if self.served[i] == 0:
                self.avgQdelay_counter[i] = 0.0
            else:
                self.avgQdelay_counter[i] = self.counter_delay[i]/self.served[i]
                self.avgQdelay_counter[i] /= 60 # convert to minutes
                self.MAX_counter_delay[i] /= 60 # convert maximum delay for each counter in minutes




        self.avg_customer_in_system = self.avg_customer_in_system/sim.simclock

        # Overall avg delay = sum(probability of a customer type * avg delay of that customer type)
        for i in range(MAX_NUM_CUST_TYPES):
            self.overall_avg_delay += sim.states.prob_dist_cust_type[i] * self.avgQdelay_cust_type[i]

        for i in range(MAX_NUM_COUNTER):
            self.avgQlength[i] = self.avgQlength[i]/sim.simclock
 



    def printResults(self, sim):
        print()

        # print ('All arrival times  - ', sim.states.queue)
        # DO NOT CHANGE THESE LINES
        print('MMk Results: lambda = ', sim.params.lambd, "service time = ", sim.params.st, "accumulated cashier time =", sim.params.act)
        print('Group probabilities :', sim.states.prob_dist_of_groups)
        print('Employee distribution ,k = ', sim.params.k)
        # print('MMk Total delay: %d' % (sim.states.total_delays))
        
        print('\nArrival_event_count- ', sim.states.arrival_event_count)
        print( 'Departure_event_count- ', sim.states.departure_event_count)
        print('Total customers served at each counter:',  sim.states.served)
        
        self.avgQdelay_cust_type = [ '%.5f' % elem for elem in self.avgQdelay_cust_type ]
        self.avgQdelay_counter = [ '%.5f' % elem for elem in self.avgQdelay_counter ]
       
        print('\nAverage queue delay for each customer type: ', self.avgQdelay_cust_type)
        print('Maximum delay for each customer type=', sim.states.MAX_cust_type_delay )

        print('\nAverage queue delay for each counter type: ', self.avgQdelay_counter)
        print('Maximum delay at each counter =', sim.states.MAX_counter_delay)
        print('Avg len of queue -',sim.states.avgQlength )
        print('Max len of q at each counter', sim.states.maxAvgQlength)

        print('\nOverall avg delay %lf' % sim.states.overall_avg_delay)
        # print('Maximum number of customers at any time instant')
        print('Avg customer in system %lf' % sim.states.avg_customer_in_system)
        
        print('Total served ', sim.states.departure_event_count)



    def getResults(self, sim):
        return (self.avgQlength, self.avgQdelay_cust_type, self.avgGroupDelay)



class Event:
    def __init__(self, sim):
        self.eventType = None
        self.sim = sim
        self.eventTime = None
        # self.groupSize = None
        # self.cust_type = None
        # self.curr_counter_task = None #current task no in routing
        

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
        
        groupSize = determine_groupSize(sim)

        arrivalTime = np.random.exponential(1/sim.params.lambd)
        for i in range(groupSize):
            cust_type = determine_cust_type(sim)
            sim.scheduleEvent(ArrivalEvent(arrivalTime, sim, 0, groupSize, cust_type, curr_counter_task=0, queueNo=0))
        
        sim.scheduleEvent(ExitEvent(int(90*60), sim))
        

class ExitEvent(Event):
    def __init__(self, eventTime, sim):
        self.eventTime = eventTime
        self.eventType = 'EXIT'
        self.sim = sim

    def process(self, sim):
        None


def determine_groupSize(sim):
    groupSize = np.random.choice(np.arange(1, 5), p=sim.states.prob_dist_of_groups)
    print("Generated group size ", groupSize)
    return groupSize

def determine_cust_type(sim):
    cust_type = np.random.choice(np.arange(1, 4), p=sim.states.prob_dist_cust_type )
    print("Generated customer type  ", cust_type)
    return cust_type



"""

If a group of size 3 comes, we create 3 separate arrivalEvent for each of the 3 customer types of that group
But we only schedule the next arrivalEvent for each group arrival - or else the arrivalEvent count becomes enormous

"""

class ArrivalEvent(Event):
    def __init__(self,eventTime, sim, groupID, groupSize, cust_type, curr_counter_task, queueNo ):
        
        self.eventType = 'ARRIVAL'
        self.eventTime = eventTime
        self.sim = sim

        self.groupID = groupID
        self.groupSize = groupSize
        self.cust_type = cust_type
        self.curr_counter_task = curr_counter_task
        self.queueNo = queueNo  #eta to track cashier counter er kon queue te customer ase

        print("Initializing ArrivalEvent at %lf" % self.eventTime, "for group", self.groupSize, 'cust_type', self.cust_type,'\n')



    def process(self, sim):
        print("in process() of ArrivalEvent")

        
        # If the customer is on the first counter of her routing then 
        # increase the number of total customers in the system.
        # schedule next group arrival event


        print('Next expected group id ', sim.states.next_expected_group_id)
        print('Current group id', self.groupID)

        if(self.groupID == sim.states.next_expected_group_id and self.curr_counter_task == 0):
            
            
            timeOfNextArrival = sim.simclock + np.random.exponential(1/sim.params.lambd)
            sim.states.next_expected_group_id =  self.groupID + 1
            
            
            # determine the group type
            next_group_type = determine_groupSize(sim)
            for i in range(next_group_type):
                cust_type = determine_cust_type(sim)

                sim.states.total_custs_in_system += 1
                sim.states.curr_cust_of_each_type[cust_type - 1] += 1
     
                sim.states.arrival_event_count_for_cust_type[self.cust_type - 1] += 1
                sim.scheduleEvent(ArrivalEvent(timeOfNextArrival, sim, sim.states.next_expected_group_id, next_group_type, cust_type, curr_counter_task=0, queueNo=0))
        
        
            #increase arrival count of this group size
            sim.states.arrival_event_count[self.groupSize - 1] +=1
            sim.states.see_counter_status(sim)
       



        # Now check if the current counter has free employees or not 
        
        counter = sim.states.route[self.cust_type][self.curr_counter_task]
        print('route[',self.cust_type,'][', self.curr_counter_task, '] =', counter)

        counterStatus, self.queueNo = sim.states.check_if_curr_counter_is_free(sim, counter) # queueNo = oi counter er kon cashier ta free
        print("Customer = ",self.cust_type, "eventTime = %lf" % self.eventTime, "Counter =", counter , "queueNo = ", self.queueNo)
        
        if (counterStatus == -1): #means the current counter is busy
    
            
            print("\n~~~~~~~~~~~~~~~~~~~~~~~ Current counter", counter ," is now busy ~~~~~~~~~~~~~~~~~~~~`")
           
            # If the current counter is the cashier then select the smallest queue.
            if(counter == CASHIER_COUNTER ):
                shortestQ = sim.states.get_shortest_Q_of_cashier_counter(sim)
                self.sim.states.queue[counter][shortestQ].append(sim.simclock)

            else:
                # put arrival time of this customer in its server's queue
                # print(sim.states.queue)
                sim.states.queue[counter][self.queueNo].append(sim.simclock)
                print('len of q[',counter, '][', self.queueNo, ']:',len(sim.states.queue[counter][self.queueNo]))
                      
            sim.states.see_counter_status(sim)




        else:

            # If counters are available then schedule a departure event. 
            # For service time, use st value if the counter is not the cashier. If cashier then this
            # will be sum of the previous act values of the food counters this customer has
            # visited, e.g: if the customer is of type sandwich then act(sandwich) + act(drinks)
            
            # Schedule a departure (service completion)
            print('service given at counter', counter, 'for customer type',  self.cust_type, '\n')
            

            sum_act = 0
            if counter == 3: #cashier counter
                for elem in sim.states.route[self.cust_type]:   #elem gives the counter no's of this route
                    if elem != CASHIER_COUNTER :                #cashier counter ta baade in route
                        lo, hi = sim.params.act[elem]
                        sum_act += np.random.uniform(lo, hi)

                
                departure_time = sim.simclock + sum_act
                sim.scheduleEvent(DepartureEvent(departure_time, sim, self.groupID, self.groupSize, self.cust_type, curr_counter_task=self.curr_counter_task,queueNo=self.queueNo))
  
            else:   
                lo, hi  = sim.params.st[counter]
                departure_time = sim.simclock + np.random.uniform(lo, hi)
                sim.scheduleEvent(DepartureEvent(departure_time, sim, self.groupID, self.groupSize, self.cust_type, curr_counter_task=self.curr_counter_task, queueNo=self.queueNo))
  
            
            sim.states.served[counter] +=1
            if counter != DRINK_COUNTER:                        # as drinks counter has no queue
                sim.states.counter_status[counter][self.queueNo] = "BUSY" 
            sim.states.see_counter_status(sim)

            
            


"""

Event function for departure of a customer from a particular counter
If the customer is not at the cashier count, this departure time is the arrival time at the next counter in route

"""
class DepartureEvent(Event):
    def __init__(self,eventTime, sim, groupID, groupSize, cust_type, curr_counter_task, queueNo):
        print("Initializing DepartureEvent")
        self.eventType = 'DEPARTURE'
        self.eventTime = eventTime
        self.sim = sim

        self.groupID = groupID
        self.groupSize = groupSize
        self.cust_type = cust_type
        self.curr_counter_task = curr_counter_task
        self.queueNo = queueNo #eta to track cashier counter er kon queue te customer ase


    def process(self, sim):
        print("in process() of DepartureEvent")

        # Find the counter where the event is happening from counter routing.
        print('cust_type', self.cust_type)
        print('curr_counter_task', self.curr_counter_task)
        counter = sim.states.route[self.cust_type][self.curr_counter_task]
        print('route[',self.cust_type,'][', self.curr_counter_task, '] =', counter)
        print('queueNo', self.queueNo)


       
        sim.states.print_queue_status(sim)
        sim.states.print_cashier_queue_status(sim)


        # Check the queue where this customer was. We have q_no for this purpose. 
        # If the queue is empty then free a counter.

        if(len(sim.states.queue[counter][self.queueNo]) == 0):

            print("The queue for counter ", counter, " is empty.\nCounter", counter, 'is made free')
            sim.states.counter_status[counter][self.queueNo] = "IDLE"


        # otherwise start processing event
        # calculate delay
        # schedule departure event for customer

        else:
            
            sim.states.see_counter_status(sim)

            # The queue is nonempty, so start service on first job in queue. 
            print("\nCounter ", counter, "queue", self.queueNo,"  is nonempty, so start service on first customer in queue ")
            
            sim.states.served[counter] += 1
            print("Customer served at counter", counter)
            
            # compute delay for this job and gather statistics
            
            # delay = sim.simclock - sim.states.queue[0][0] #curr time - time of arrival of this customer who is now being served
            delay_ = sim.simclock - sim.states.queue[counter][self.queueNo][0]
            sim.states.counter_delay[counter] += delay_
            sim.states.customer_type_delay[self.cust_type-1] += delay_

            # max delay for each customer type and counter
            sim.states.MAX_counter_delay[counter] = max(sim.states.MAX_counter_delay [counter], delay_ )
            sim.states.MAX_cust_type_delay [ self.cust_type - 1] = max( sim.states.MAX_cust_type_delay[self.cust_type - 1], delay_)


            if counter != CASHIER_COUNTER :
                # schedule a departure event for this job
                print("Scheduling a departure event for group id ",self.groupID ,"customer type ", self.cust_type)
                
                lo, hi  = sim.params.st[counter]
                departure_time = sim.simclock + np.random.uniform(lo, hi)
                sim.scheduleEvent(DepartureEvent(departure_time, sim, self.groupID, self.groupSize, self.cust_type, curr_counter_task=self.curr_counter_task, queueNo=self.queueNo))

            
            
            # remove the first customer from queue
            sim.states.queue[counter][self.queueNo] = sim.states.queue[counter][self.queueNo][1:]
          
        # Now check if the customer has reached the final counter in her routing. 
        # If no then schedule another arrival event, where the event time will be the event time of the
        # current departure event. The rest will be the same except for the current counter
        # index. This will be incremented by 1. 
        # If the customer has reached the last counter
        # of her routing then decrease the total number of customers in the system.
        
    
        if(counter == CASHIER_COUNTER): #means customer is in final station
            
            print("Customer is now in cashier counter")
            sim.states.curr_cust_of_each_type [self.cust_type - 1] -=1
            sim.states.departure_event_count +=1
            return

        else:
        
            print("Customer is not at cashier's yet so move customer to next counter on its route")

            print("Increased counter task by 1")
            self.curr_counter_task += 1;[self.queueNo]
            
            print("Scheduling ArrivalEvent to next counter - cust_type", self.cust_type, "curr_counter_task", self.curr_counter_task,"\n")
            sim.scheduleEvent(ArrivalEvent(sim.simclock, sim, self.groupID, self.groupSize , self.cust_type, self.curr_counter_task, self.queueNo))

       


class Simulator:
    def __init__(self, seed):
        print("in __init__ of Simulator")
        self.eventQ = []
        self.simclock = 0
        self.seed = seed
        self.params = None
        self.states = None

    def initialize(self):
        print("Initializing Simulator")
        self.simclock = 0
        self.states.set_counter_and_queue_status(self)  #-- changed here
        
        self.scheduleEvent(StartEvent(0, self))

    def configure(self, params, states):
        print("Configuring Simulator")
        self.params = params
        self.states = states
        self.params.print__init__(self)
        self.states.print__init__(self)

    def now(self):
        print("in now of Simulator")
        return self.simclock

    def scheduleEvent(self, event):
        # print("in scheduleEvent of Simulator")
        heapq.heappush(self.eventQ, (event.eventTime, event))

    def run(self):
        print("Running Simulator")
        random.seed(self.seed)
        # np.random.seed(self.seed)

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
            if event.eventType != 'START':
                print('              At %lf' % event.eventTime, 'Event', event, 'GroupID', event.groupID, 'custType', event.cust_type, 'k =', self.params.k,'\n')
            else:
                print('              At %lf' % event.eventTime, 'Event', event, 'k = ', self.params.k,'\n')
                    
            self.simclock = event.eventTime
            event.process(self)

        self.states.finish(self)

    def printResults(self):
        self.states.printResults(self)

    def getResults(self):
        print("in getResults of Simulator")
        return self.states.getResults(self)



def cafeteria():

    print("Starting cafeteria simulation\n")
    seed = 100

    # --------------- hardcoded input -----------
    # base case - 4 employees
    k = [1, 1, 2] #number of employees at each counter - a list

    # 5 employees
    # k = [1, 1, 3]
    # k = [2, 1, 2]
    # k = [1, 2, 2]

    # # 6 employees
    k = [2, 2, 2]
    # k = [2, 1, 3]
    # k = [1, 2, 3]

    # # 7 employees
    # k = [2, 2, 3]

    prob_dist_of_groups = [0.5, 0.3, 0.1, 0.1] 
    mean = 30                                                           #Inter-arrival times between groups are exponentially distributed with mean 30 seconds
    lambd = 1.0/mean
    prob_dist_cust_type = [0.8, 0.15, 0.05]

    route = {1:[0,2,3], 2:[1,2,3], 3:[2,3]}                             # 1- indexed



    service_time = [(50/k[0], 120/k[0]), (60/k[1], 180/k[1]), (5, 20)]  # increase of employees in food counters decreases the counter's service time, but food counters always have one queue and drinks counter has no queue
    accumulated_cashier_time = [(20/k[0], 40/k[0]), (5/k[1], 15/k[1]), (5, 10)]
    cashierCount = k[2]

    sim = Simulator(seed)
    states = States(route, prob_dist_of_groups, prob_dist_cust_type, cashierCount)
    sim.configure(Params(lambd, st=service_time, act=accumulated_cashier_time, k=k), states)
    # sim.initialize()
    sim.run()
    sim.printResults()


def main():
    cafeteria()

if __name__ == "__main__":
    main()
