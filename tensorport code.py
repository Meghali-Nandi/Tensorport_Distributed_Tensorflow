import tensorflow as tf
from tensorport import get_data_path, get_logs_path


# Get the environment parameters for distributed TensorFlow
try:
    job_name = os.environ['JOB_NAME']
    task_index = os.environ['TASK_INDEX']
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except: # we are not on TensorPort, assuming local, single node
    task_index = 0
    ps_hosts = None
    worker_hosts = None
        
        
# This function defines the master, ClusterSpecs and device setters
def device_and_target():
    # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
    # Don't set a device.
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")

    # Otherwise we're running distributed TensorFlow.
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
            "ps": FLAGS.ps_hosts.split(","),
            "worker": FLAGS.worker_hosts.split(","),
    })
    server = tf.train.Server(
            cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
            tf.train.replica_device_setter(
                    worker_device=worker_device,
                    cluster=cluster_spec),
            server.target,
    )

    device, target = device_and_target()        

# Defining graph   
with tf.device(device):
        #TODO define your graph here
        ...

#Defining the number of training steps
hooks=[tf.train.StopAtStepHook(last_step=100000)]

with tf.train.MonitoredTrainingSession(master=target,
    is_chief=(FLAGS.task_index == 0),
    checkpoint_dir=FLAGS.logs_dir,
    hooks = hooks) as sess:

    while not sess.should_stop():
            # execute training step here (read data, feed_dict, session)
            # TODO define training ops
            data_batch = ...
            feed_dict = {...}
            loss, _ = sess.run(...)