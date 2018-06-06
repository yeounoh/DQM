import numpy as np
import argparse, os, sys, json
import math
import simpleamt
from boto.mturk.price import Price
from boto.mturk.question import HTMLQuestion
from boto.mturk.connection import MTurkRequestError
import io, time


def process_assignments(mtc, hit_id, status):
    results = []
    page_number = 1
    while True:
        try:
            assignments = mtc.get_assignments(hit_id,
                            page_number=page_number, page_size=100)
            if len(assignments) == 0:
                return results
        except:
            print >> sys.stderr, ('Bad hit_id %s' % str(hit_id))
            return results
        for a in assignments:
            if a.AssignmentStatus in status:
                try:
                    output = json.loads(a.answers[0][0].fields[0])
                except ValueError as e:
                    print >> sys.stderr, ('Bad data from assignment %s (worker %s)'
                                % (a.AssignmentId, a.WorkerId))
                    mtc.reject_assignment(a.AssignmentId, feedback='Invalid results')
                    continue
                results.append({
                    'assignment_id': a.AssignmentId,
                    'hit_id': hit_id,
                    'worker_id': a.WorkerId,
                    'output': json.loads(a.answers[0][0].fields[0]),
                    'submit_time': a.SubmitTime,
                })
            page_number += 1
    return results

def disable_hit(hit_ids, sandbox=True):
    parser = argparse.ArgumentParser(parents=[simpleamt.get_parent_parser()],
              description="Delete HITs")
    args = parser.parse_args()
    args.sandbox = sandbox
    mtc = simpleamt.get_mturk_connection_from_args(args)

    for hit_id in hit_ids:
        print ('This will delete HIT with ID: %s with sandbox=%s'
             % (hit_id, str(args.sandbox)))
        try:
            mtc.disable_hit(hit_id)
        except:
            print 'Failed to disable: %s' % (hit_id)
        else:
            print 'Aborting'

def create_batch(records, batch_size=10):
    indices = np.random.randint(len(records), size=batch_size)
    return np.array(records)[indices]

def launch_hit(hit_template_path, hit_properties_path, hit_ids_path, sandbox=True):
    parser = argparse.ArgumentParser(parents=[simpleamt.get_parent_parser()])
    args = parser.parse_args()
    args.sandbox = sandbox

    mtc = simpleamt.get_mturk_connection_from_args(args)
    with open(hit_properties_path, 'r') as hit_properties_file:
        hit_properties = json.load(hit_properties_file)
    hit_properties['reward'] = Price(hit_properties['reward'])
    simpleamt.setup_qualifications(hit_properties, mtc)

    frame_height = hit_properties.pop('frame_height')
    env = simpleamt.get_jinja_env(args.config)
    template = env.get_template(hit_template_path)

    hit_ids = []
    with open(hit_ids_path,'w') as hit_ids_file:
        template_params = {'input': ''}
        html = template.render(template_params)
        html_question = HTMLQuestion(html, frame_height)
        hit_properties['question'] = html_question

        launched = False
        while not launched:
            try:
                boto_hit = mtc.create_hit(**hit_properties)
                launched = True
            except MTurkRequestError as e:
                print e
        hit_id = boto_hit[0].HITId
        hit_ids_file.write('%s\n' % hit_id)
        print 'Launched HIT ID: %s' % (hit_id)

        hit_ids.append(hit_id)

    return hit_ids

def get_results(hit_ids, sandbox=True):
    parser = argparse.ArgumentParser(parents=[simpleamt.get_parent_parser()])
    args = parser.parse_args()
    args.sandbox = sandbox
    mtc = simpleamt.get_mturk_connection_from_args(args)

    results = []
    status = ['Approved', 'Submitted']

    for hit_id in hit_ids:
        prev_size = len(results)
        start_time = time.time()
        while len(results) <= prev_size and (time.time()-start_time) <= 800:
            results += process_assignments(mtc, hit_id, status)
            time.sleep(5)

    return results    

def triangular_walk_using_amt(data, label, update_template, 
                              hit_id_file_path, template_file_path, 
                              property_file_path, result_file_path,
                              n_workers=1000, n_max=50, sandbox=True, dirty_is_no=True):
    n_items = len(data)

    walks = dict() # walks[i] = (record_id, n, k)
    completed = set() # keeps track of completed triangles for batch item replacement
    estimates = dict() # estimates[n_worker_] = #error_estimate
    linear_estimates = dict()

    batch_size = 20 # this is fixed
    for i in range(batch_size):
        walks[i] = (np.random.choice(n_items), 0., 0.)
        linear_estimates[i] = list()
    batch = []
    for k, v in walks.iteritems():
        batch.append(data[v[0]])

    n_workers_ = n_workers
    while n_workers_ > 0:
        update_template(batch)
        try: os.remove(hit_id_file_path)
        except OSError: pass

        hit_ids = launch_hit(sandbox=sandbox,
                    hit_template_path=template_file_path,
                    hit_properties_path=property_file_path,
                    hit_ids_path=hit_id_file_path) # post a random batch task

        hits = get_results(hit_ids,sandbox=sandbox) # for now, results is a singleton list
        with open(result_file_path,'a+') as f:
            for hit in hits:
                hit['batch'] = batch
                f.write(json.dumps(hit)+'\n')
        disable_hit(hit_ids,sandbox=sandbox) # remove the HITs on hit_ids.txt
        print 'n_workers %s'%(n_workers-n_workers_+1)

        if len(hits) == 0:
            continue

        # this allows multiple assignments per hit
        # if we restrict a single assignment per hit, then we can pretend that the for loop is not here.
        for hit in hits:
            for i in range(batch_size):
                record_id = walks[i][0]
                l_ = label[data[record_id]]
                n_ = walks[i][1] + 1.
                k_ = walks[i][2]

                resp = hit['output']['v%s_is_valid'%(i+1)]
                if dirty_is_no:
                    if resp == 'no':
                        k_ += 1
                else:
                    if resp == 'yes':
                        k_ += 1

                # check for stopping conditions
                if n_ < n_max and k_/n_ > 0.5:
                    walks[i] = (walks[i][0], n_, k_)
                else:
                    completed.add(i)
                    if k_/n_ <= 0.5:
                        linear_estimates[i].append(0.)
                    else:
                        if (2-n_max-2*k_)**2 -4*(2*n_max-2)*k_ >= 0:
                            p_ = ( 2.*k_+n_max-2+math.sqrt((2-n_max-2*k_)**2-4*(2*n_max-2)*k_)) / (4.*n_max-4)
                        else:
                            p_ = ((2.*k_+n_max-2)/(4*n_max-4))
                        #p_ = k_/n_max
                        #print 1./(2*p_-1.)
                        p_ = max(p_, 0.6)
                        linear_estimates[i].append(1./(2*p_-1.) - 0.165*p_*(1-p_)/(2*p_ - 1)**2)

            for i in completed:
                walks[i] = (np.random.choice(n_items), 0., 0.)
                batch[i] = data[walks[i][0]]

            cur_estimates = list()

            for i in range(batch_size):
                if len(linear_estimates[i]) > 0:
                    cur_estimates.append(np.mean(linear_estimates[i]))

            if len(cur_estimates) > 0:
                estimates[n_workers-n_workers_+1] = np.mean(cur_estimates) * n_items
            n_workers_ -= 1
            completed = set()

    return estimates

