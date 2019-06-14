import caffe
import h5py
import random
import numpy as np
import time
import gc
import sys
# MTDataLayer parmas example:
# {
#     "sources":[
#         {"source":"123.txt", "batch_size":64, "data_shape":(3,112,112),"label_shape":(3,112,112)},
#         {"source":"456.txt", "batch_size":32, "data_shape":(3,112,112),"label_shape":(2)}
#     ],
#     "shuffle":True
# }


# "{\"sources\":[{\"source\":\"landmark_hdf5.txt\", \"batch_size\":64, \"data_shape\":(3,112,112), \"label_shape\":(254)},{\"source\":\"eyelid_hdf5.txt\", \"batch_size\":32, \"data_shape\":(3,112,112), \"label_shape\":(2)}], \"shuffle\":True}"

#multiply task data layer
class MTDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.batch_loader = MTBatchLoader(params)
        # self.batch_size = self.batch_loader.get_batch_size()
        shape_list = self.batch_loader.get_shapes()
        for i,shape in enumerate(shape_list):
            if len(shape) == 1:
                top[i].reshape(int(shape[0]))
            elif len(shape) == 2:
                top[i].reshape(int(shape[0]), int(shape[1]))
            elif len(shape) == 3:
                top[i].reshape(int(shape[0]), int(shape[1]), int(shape[2]))
            elif len(shape) == 4:
                top[i].reshape(int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
            else:
                top[i].reshape(shape)   # type error will occur for this statement
        
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):

        batch_list = self.batch_loader.next()
        for i,data_or_label in enumerate(batch_list):
            top[i].data[...] = data_or_label

    def backward(self, top, propagate_down, bottom):
        pass

    
class ValidLayer(caffe.Layer):
    #bottom: include one or more task predictions, one or more task labels, one valid_index.For example bottom: task1_data, task2_data, task1_label, task2_label, invalid_index
    #top: one or more task invalid data, for example, task1_valid_data, task2_valid_data
    def setup(self, bottom, top):
        params = eval(self.param_str)
        label_shapes = params["label_shapes"]
        batch_size = params["real_batch_size"]
        if len(bottom) < 3 :
            raise Exception("Need at least 3 inputs.")
        elif len(bottom)%2==0:
            raise Exception("Input number should be odd.")

        self.task_num = (len(bottom) - 1)//2
    
        for i,shape in enumerate(label_shapes):
            if isinstance(shape, np.int64) or isinstance(shape, np.int32):
                top[i].reshape(int(batch_size), int(shape))
            elif len(shape) == 1:
                top[i].reshape(int(batch_size), int(shape[0]))
            elif len(shape) == 2:
                top[i].reshape(int(batch_size), int(shape[0]), int(shape[1]))
            elif len(shape) == 3:
                top[i].reshape(int(batch_size), int(shape[0]), int(shape[1]), int(shape[2]))
            else:
                top[i].reshape([batch_size]+list(shape))     # type error will occur for this statement

        self.valid_index_list = []
        self.invalid_index_list = []
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        
        for i in range(self.task_num):
            valid_index = np.where(bottom[-1].data == i)
            invalid_index = np.where(bottom[-1].data != i)
            self.valid_index_list.append(valid_index)
            self.invalid_index_list.append(invalid_index)

        for i in range(self.task_num):
            
            top[i].data[...] = bottom[i].data
            invalid_index = self.invalid_index_list[i]
            # assign label to top, so that the loss between them is 0.
            top[i].data[invalid_index,...] = bottom[i+self.task_num].data[invalid_index,...]    

    def backward(self, top, propagate_down, bottom):
        for i in range(self.task_num):
            if  propagate_down[i] :
                valid_index = self.valid_index_list[i]
                bottom[i].diff[...] = 0
                bottom[i].diff[valid_index] = top[i].diff[valid_index]
                

class BatchIterator(object):
 
    def __init__(self, source, batch_size, data_shape, label_shape,shuffle=True, loop=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loop = loop
        self.datalist = open(source,"r").read().splitlines()
        example = h5py.File(self.datalist[0])
        assert data_shape == example['data'][0].shape, "data_shape:{} != example['data'][0].shape:{}".format(data_shape , example['data'][0].shape)
        assert label_shape == example['label'][0].shape, "label_shape:{} != example['label'][0].shape:{}".format(label_shape, example['label'][0].shape)
        self.len_example = len(example['data'])
        example.close()

        self.file_num = len(self.datalist)
        self.data_shape = data_shape
        self.label_shape = label_shape

        self.index_list = list(range(0, self.file_num, self.batch_size))
        self.i = 0
        
        if self.shuffle:
            self.do_shuffle()
        gc.collect()

        print("BatchIterator.__init__()")

    def __iter__(self):
        return self
    
    def num_per_file(self):
        return self.len_example

    def do_shuffle(self):
        random.shuffle(self.datalist)

    def __next__(self):
        # t1 = time.time()
        
        if self.i >= len(self.index_list):
            if self.loop:
                self.i = 0
                if self.shuffle:
                    self.do_shuffle()
            else:
                raise StopIteration
            
        index = self.index_list[self.i]
        self.i = self.i + 1

        pad = index + self.batch_size - self.file_num
        pad = pad if pad>0 else 0
        # batch hdf5 file names
        if pad > 0:
            batch = self.datalist[index:index+self.batch_size] + random.sample(self.datalist[0:index], pad)
        else:
            batch = self.datalist[index:index+self.batch_size]

        tmp = [self.batch_size*self.len_example]+list(self.data_shape)
        batch_data = np.empty(tmp, dtype=np.float32)
        tmp = [self.batch_size*self.len_example]+list(self.label_shape)
        batch_label = np.empty(tmp, dtype=np.float32)

        for j,f in enumerate(batch):
            example = h5py.File(f, "r")
            index = j*self.len_example
            end_index = index + self.len_example

            # example['data'].read_direct(batch_data[ index:end_index,...])
            # example['label'].read_direct(batch_label[index:end_index,...])

            batch_data[ index:end_index,...] = np.array(example['data'])
            batch_label[index:end_index,...] = np.array(example['label'])
            
            example.close()
        
        # t2 = time.time()
        # print("BatchIterator.__next__() use time: {} ms".format((t2-t1)*1000))

        return batch_data, batch_label

    def next(self):  #for python2
        return self.__next__()                

class MTBatchLoader(object):
    def __init__(self, params):
        sources = params["sources"]
        self.shuffle = params["shuffle"]

        self.source_num = len(sources)
        self.batch_iter_list = []
        data_shape_list = []
        self.label_shape_list = []
        self.batch_size_list = []
        self.num_per_file_list = []
        for s in sources:
            iter = BatchIterator(s["source"], s["batch_size"], tuple(s["data_shape"]), tuple(s["label_shape"]))
            self.batch_iter_list.append(iter)
            self.num_per_file_list.append(iter.num_per_file())
            data_shape_list.append(tuple(s["data_shape"]))
            self.label_shape_list.append(tuple(s["label_shape"]))
            self.batch_size_list.append(s["batch_size"])
        
        # self.real_batch_size = np.sum(self.batch_size_list)
        self.real_batch_size = np.dot(self.num_per_file_list, self.batch_size_list)
        self.data_shape = data_shape_list[0]

        self.real_batch_shape_list = self.get_shapes()
        
        self.batch_data = np.empty(self.real_batch_shape_list[0], dtype=np.float32)

        self.multi_task_labels = []

        for shape in self.real_batch_shape_list[1: -1]:
            task_label = np.empty(shape, dtype=np.float32)
            self.multi_task_labels.append(task_label)

        self.valid_index = np.empty(self.real_batch_shape_list[-1], dtype=np.float32)

        self.empty_cnt = 0

        print("MTBatchLoader.__init__()")

    def get_batch_size(self):
        return self.real_batch_size

    def get_shapes(self):
        # add data shape
        shape_list = [self.data_shape]+self.label_shape_list
        batch_shape_list = []
        for shape in shape_list:
            tmp = tuple([self.real_batch_size]+list(shape))
            batch_shape_list.append(tmp)

        # append valid_index_shape
        valid_index_shape = tuple([self.real_batch_size])
        batch_shape_list.append(valid_index_shape)
        return batch_shape_list

    def __iter__(self):
        return self

    def __next__(self):
        # t1 = time.time()
        gc.disable()
        if self.empty_cnt >= len(self.label_shape_list) :
            # raise StopIteration
            self.empty_cnt = 0

        index = 0           
        for i,iter in enumerate(self.batch_iter_list):
            try:
                datas, labels = iter.next()
                end_index = index+len(datas)
                
                self.batch_data[index: end_index] = datas
                self.multi_task_labels[i][index: end_index] = labels
                self.valid_index[index: end_index] = np.ones(self.batch_size_list[i]*self.num_per_file_list[i])*i
            except Exception as e:  # no data for task i
                tmp = [self.batch_size_list[i]*self.num_per_file_list[i]] + list(self.data_shape)
                datas = np.zeros(tmp, dtype=np.float32)
                end_index = index+len(datas)
                self.batch_data[index: end_index] = datas

                tmp = [self.batch_size_list[i]*self.num_per_file_list[i]] + list(self.label_shape_list[i])
                labels = np.zeros(tmp, dtype=np.float32)
                self.multi_task_labels[i][index: end_index] = labels
                
                tmp = [self.batch_size_list[i]*self.num_per_file_list[i]].append(len(self.label_shape_list))
                self.valid_index[index: end_index] = np.ones(self.batch_size_list[i]*self.num_per_file_list[i])*(-1)

                self.empty_cnt = self.empty_cnt + 1
                print("Exception:{}".format(e))

            finally:
                index = index + len(datas)
            
        ret = []
        if self.shuffle:
            sample_index =  list(range(self.real_batch_size)) 
            random.shuffle(sample_index)

            ret_batch_data = self.batch_data[sample_index]
            ret.append(ret_batch_data)
            
            for task_label in self.multi_task_labels:
                ret.append( task_label[sample_index] )

            ret_valid_index = self.valid_index[sample_index]
            ret.append(ret_valid_index)
            
        else:
            ret.append(self.batch_data)
            for task_label in self.multi_task_labels:
                ret.append(task_label)
            ret.append(self.valid_index)
        
        gc.enable()
        # t2 = time.time()
        # print("MTBatchLoader.__next__() use time: {} ms".format((t2-t1)*1000))
        return ret

    def next(self):
        return self.__next__()

        
