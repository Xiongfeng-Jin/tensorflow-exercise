import math
import tempfile
import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir','/tmp/mnust-data','Directory for storing mnist data')
flags.DEFINE_integer('hidden_units',100,'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps',1000000,'Number of (global) training steps to perform')
flags.DEFINE_integer('batch_size',100,'Training batch size')
flags.DEFINE_float('learning_rate',0.01,'Learning rate')
flags.DEFINE_boolean('sync_replicas',False,'Use the sync_replicas (synchronized replicas) mode, wherein the parameter updates from workers are aggregated before applied to avoid stale gradients')
flags.DEFINE_integer('replicas_to_aggregate',None,'Number of replicas to aggregate before parameter update is applied (For sync_repliacs mode only; default: num_workers)')
flags.DEFINE_string('ps_hosts','192.168.2.101:6666','Comma-separated list of hostname:port pairs')
flags.DEFINE_string('worker_hosts','192.168.2.100:6666,192.168.2.100:6667','Comma-separated list of hostname:port pairs')
flags.DEFINE_string('job_name',None,'job name: worker or ps')
flags.DEFINE_integer('task_index',None,'Worker task index, should be >= 0. task_index=0 is the master worker task the performs the variable initialization')

FLAGS = flags.FLAGS
IMAGE_PIXELS = 28

def main(unused_argv):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	if FLAGS.job_name is None or FLAGS.job_name == '':
		raise ValueError('Must specify an explicit job name')
	if FLAGS.task_index is None or FLAGS.task_index == '':
		raise ValueError('Must specify an explicit task_index')
	print('job name = %s ' FLAGS.job_name)
	print('task index %d ' % FLAGS.taks_index)
	
	ps_spec = FLAGS.ps_hosts.split(',')
	worker_spec = FLAGS.worker_hosts.split(',')
	num_workers = len(worker_spec)
	cluster = tf.train.ClusterSpec({"ps":ps_spec,"worker":worker_spec})
	server = tr.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
	if FLAGS.job_name == 'ps':
		server.join()
	#not finished
	
	
	

if __name__ == "__main__":
	main()