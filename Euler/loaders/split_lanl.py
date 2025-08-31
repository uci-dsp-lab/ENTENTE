import os
import pickle
from tqdm import tqdm

# LANL is so huge it's prohibitively expensive to scan it for
# edges of later time steps. To remedy this (and make it easier
# for the distro models to load data later on) I have split it into
# files containing 10,000 seconds each

# Please obtain the LANL data set from:
# https://csr.lanl.gov/data/cyber1/

# RED = '' # Location of redteam.txt
# SRC = '' # Location of auth.txt
# DST = '' # Directory to save output files to

# SRC_DIR = '' # Directory of dns.txt, flows.txt, proc.txt, auth.txt


# assert RED and SRC and DST, 'Please download the LANL data set, and mark in the code where it is:\nLines 13-15 of /lanl_experiments/loaders/split.py'

# DELTA = 10000
# DAY = 60**2 * 24

def mark_anoms():
    '''
    Parses the redteam file and creates a small dict of
    nodes involved with anomalous edges, and when they occur
    '''
    with open(RED, 'r') as f:
        red_events = f.read().split()

    # Slice out header
    red_events = red_events[1:]

    def add_ts(d, val, ts):
        # val = (src_u, src, dst)
        val = (val[1], val[2])
        if val in d:
            d[val].append(ts)
        else:
            d[val] = [ts]

    anom_dict = {}
    for event in red_events:
        tokens = event.split(',')
        ts = int(tokens.pop(0))
        add_ts(anom_dict, tokens, ts)

    return anom_dict


def mark_anoms_node():
    '''
    Parses the redteam file and creates a small dict of
    nodes involved with anomalous edges, and when they occur
    '''
    with open(RED, 'r') as f:
        red_events = f.read().split()

    # Slice out header
    red_events = red_events[1:]

    def add_ts(d, val, ts):
        # val = (src_u, src, dst)
        if val[1] in d:
            d[val[1]].append(ts)
        else:
            d[val[1]] = [ts]
        if val[2] in d:
            d[val[2]].append(ts)
        else:
            d[val[2]] = [ts]

    anom_dict = {}
    for event in red_events:
        tokens = event.split(',')
        ts = int(tokens.pop(0))
        add_ts(anom_dict, tokens, ts)

    return anom_dict


def is_anomalous(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False

    times = d[(src,dst)]
    for time in times:
        # Mark true if node appeared in a comprimise
        # in the last 24 hrs (as was done by Nethawk)
        # ZL: i don't see where 24 hrs checking is done, it's just ts matching
        if ts == time:
            return True

    return False

#comparing ts to time -/+ 5min
def is_anomalous_range(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False

    times = d[(src,dst)]
    for time in times:
        # Mark true if node appeared in a compromise in -/5min
        if abs(ts-time) <= 300:
            return True

    '''
    #Sanity check, disregard time
    if (src,dst) in d:
        return True
    '''

    return False

#comparing ts to time -/+ 5min, only one node is used
def is_anomalous_node_range(d, node, ts):
    if ts < 150885 or node not in d:
        return False

    times = d[node]
    for time in times:
        # Mark true if node appeared in a compromise in -/5min
        if abs(ts-time) <= 300:
            return True

    return False

def save_map(m, fname):
    m_rev = [None] * (max(m.values()) + 1)
    for (k,v) in m.items():
        m_rev[v] = k

    #with open(DST + 'nmap.pkl', 'wb+') as f:
    with open(DST + fname, 'wb') as f:
        pickle.dump(m_rev, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(DST + fname + ' saved')

def get_or_add(n, m, id):
    if n not in m:
        m[n] = id[0]
        id[0] += 1

    return m[n]


def split_auth():
    anom_dict = mark_anoms()

    last_time = 1
    cur_time = 0

    f_in = open(SRC,'r')
    #f_out = open(DST + str(cur_time) + '.txt', 'w+')
    f_out = open(DST + str(cur_time) + '.txt', 'w')

    line = f_in.readline() # Skip headers
    line = f_in.readline()

    nmap = {}
    nid = [0]
    umap = {}
    uid = [0]
    dmap = {}
    did = [0]
    atmap = {}
    atid = [0]
    ltmap = {}
    ltid = [0]
    aomap = {}
    aoid = [0]
    smap = {}
    sid = [0]
    #zl: add other edge features

    prog = tqdm(desc='Seconds parsed', total=5011199)

    fmt_domain = lambda x : x.split('@')[1]

    fmt_u = lambda x : \
        x.split('@')[0].replace('$', '')

    fmt_label = lambda ts,src,dst : \
        1 if is_anomalous(anom_dict, src, dst, ts) \
        else 0

    # Really only care about time stamp, and src/dst computers
    # Hopefully this saves a bit of space when replicating the huge
    # auth.txt flow file
    fmt_line = lambda ts,src,dst,src_d,dst_d,auth_t,logon_t,auth_o,success, src_u, dst_u: (
        '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
            ts, get_or_add(src, nmap, nid), get_or_add(dst, nmap, nid),
            get_or_add(fmt_domain(src_d), dmap, did), get_or_add(fmt_domain(dst_d), dmap, did),
            get_or_add(auth_t, atmap, atid), get_or_add(logon_t, ltmap, ltid),
            get_or_add(auth_o, aomap, aoid), get_or_add(success, smap, sid),
            fmt_label(int(ts),src,dst),get_or_add(fmt_u(src_u), umap, uid), get_or_add(fmt_u(dst_u), umap, uid)
        ),
        int(ts)
    )

    while line:
        # Some filtering for better FPR/less Kerb noise
        if 'NTLM' not in line.upper():
            line = f_in.readline()
            continue

        tokens = line.split(',')
        # print(tokens)
        #0: ts, 1: src_user_domain, 2: dest_user_domain, 3: src_c, 4: dest_c, 5:auth_type, 6: logon_type, 7: auth_orientation, 8: success/failure
        # last field has '\n', need to be removed
        #ZL: a big fixed, [:-1] should be put after tokens[8] rather than tokens[2], the result shouldn't be changed as we don't use tokens[2] and tokens[8]
        l, ts = fmt_line(tokens[0], tokens[3], tokens[4], tokens[1], tokens[2], tokens[5], tokens[6], tokens[7], tokens[8][:-1], tokens[1], tokens[2])

        if ts != last_time:
            prog.update(ts-last_time)
            last_time = ts

        # After ts progresses at least 10,000 seconds, make a new file
        if ts >= cur_time+DELTA:
            f_out.close()
            #print(DST + str(cur_time) + '.txt saved')
            cur_time += DELTA
            #f_out = open(DST + str(cur_time) + '.txt', 'w+')
            f_out = open(DST + str(cur_time) + '.txt', 'w')

        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(nmap, 'nmap.pkl')
    save_map(umap, 'umap.pkl')
    save_map(dmap, 'dmap.pkl')
    save_map(atmap, 'atmap.pkl')
    save_map(ltmap, 'ltmap.pkl')
    save_map(aomap, 'aomap.pkl')
    save_map(smap, 'smap.pkl')

def reverse_load_map(fname):
    #mapping pickle is a list, need to reverse it to a dict
    m = {}

    with open(fname, 'rb') as f:
        l = pickle.load(f)
        for i in range(0, len(l)):
            m[l[i]] = i
    # print(m)
    # exit()
    return m

def split_dns():
    anom_dict = mark_anoms()

    last_time = 1
    cur_time = 0

    f_in = open(SRC_DIR+'dns.txt','r')
    #f_out = open(DST + str(cur_time) + '.txt', 'w+')
    dns_folder = DST + 'DNS/'
    if not os.path.exists(dns_folder):
        os.makedirs(dns_folder)
        print(dns_folder + " is created!")

    f_out = open(dns_folder + str(cur_time) + '.txt', 'w')

    #line = f_in.readline() # Skip headers
    line = f_in.readline()

    nmap = reverse_load_map(DST+'nmap.pkl')

    #the total is read from the last line
    prog = tqdm(desc='Seconds parsed', total=5011199)

    fmt_label = lambda ts,src,dst : \
        1 if is_anomalous_range(anom_dict, src, dst, ts) \
        else 0

    # Really only care about time stamp, and src/dst computers
    # Hopefully this saves a bit of space when replicating the huge
    # auth.txt flow file
    fmt_line = lambda ts,src,dst: (
        '%s,%s,%s,%s\n' % (
            ts, nmap[src], nmap[dst],
            fmt_label(int(ts),src,dst)
        ),
        int(ts)
    )

    while line:
        # Filtering out the rows with ?, likely data collection errors
        if '?' in line:
            line = f_in.readline()
            continue

        tokens = line.split(',')

        #Filtering out the lines with src and dest that are not in auth
        if not tokens[1] in nmap or not tokens[2][:-1] in nmap:
            line = f_in.readline()
            continue

        #0: ts, 1: src_c, 2: dest_c
        # last field has '\n', need to be removed
        l, ts = fmt_line(tokens[0], tokens[1], tokens[2][:-1])

        if ts != last_time:
            prog.update(ts-last_time)
            last_time = ts

        # After ts progresses at least 10,000 seconds, make a new file
        if ts >= cur_time+DELTA:
            f_out.close()
            cur_time += DELTA
            f_out = open(dns_folder + str(cur_time) + '.txt', 'w')

        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    #not updating node map to make sure only nodes in auth are considered
    #save_map(nmap, 'nmap.pkl')


def split_flows():
    anom_dict = mark_anoms()

    last_time = 1
    cur_time = 0

    f_in = open(SRC_DIR+'flows.txt','r')
    #f_out = open(DST + str(cur_time) + '.txt', 'w+')
    flows_folder = DST + 'flows/'
    if not os.path.exists(flows_folder):
        os.makedirs(flows_folder)
        print(flows_folder + " is created!")

    f_out = open(flows_folder + str(cur_time) + '.txt', 'w')

    #line = f_in.readline() # Skip headers
    line = f_in.readline()

    nmap = reverse_load_map(DST+'nmap.pkl')

    #port mapping
    port_map = {}
    port_id = [0]
    #protocol mapping
    proto_map = {}
    proto_id = [0]

    #the total is read from the last line
    prog = tqdm(desc='Seconds parsed', total=3126928)

    fmt_label = lambda ts,src,dst : \
        1 if is_anomalous_range(anom_dict, src, dst, ts) \
        else 0

    #0: ts, 1: duration, 2: source computer, 3: source port, 4: destination computer, 5: destination port, 6: protocol, 7: packet count, 8: byte count
    fmt_line = lambda ts,duration,src,src_p,dst,dst_p,proto,pkt_cnt,byte_cnt: (
        '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
            ts, nmap[src], nmap[dst],get_or_add(src_p, port_map, port_id),
            get_or_add(dst_p, port_map, port_id), get_or_add(proto, proto_map, proto_id),
            duration, pkt_cnt, byte_cnt, fmt_label(int(ts),src,dst)
        ),
        int(ts)
    )

    while line:
        # Filtering out the rows with ?, likely data collection errors
        if '?' in line:
            line = f_in.readline()
            continue

        tokens = line.split(',')

        #Filtering out the lines with src and dest that are not in auth
        if not tokens[2] in nmap or not tokens[4] in nmap:
            line = f_in.readline()
            continue

        # last field has '\n', need to be removed
        l, ts = fmt_line(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7], tokens[8][:-1])

        if ts != last_time:
            prog.update(ts-last_time)
            last_time = ts

        # After ts progresses at least 10,000 seconds, make a new file
        # ZL: the last snapshot isn't saved
        if ts >= cur_time+DELTA:
            f_out.close()
            cur_time += DELTA
            f_out = open(flows_folder + str(cur_time) + '.txt', 'w')

        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(port_map, 'pomap.pkl')
    save_map(proto_map, 'prmap.pkl')


def split_proc():
    anom_dict = mark_anoms_node()

    last_time = 1
    cur_time = 0

    f_in = open(SRC_DIR+'proc.txt','r')
    #f_out = open(DST + str(cur_time) + '.txt', 'w+')
    proc_folder = DST + 'proc/'
    if not os.path.exists(proc_folder):
        os.makedirs(proc_folder)
        print(proc_folder + " is created!")

    f_out = open(proc_folder + str(cur_time) + '.txt', 'w')

    #line = f_in.readline() # Skip headers
    line = f_in.readline()

    nmap = reverse_load_map(DST+'nmap.pkl')
    umap = reverse_load_map(DST+'umap.pkl')

    #port mapping
    proc_map = {}
    proc_id = [0]

    #status mapping
    status_map = {}
    status_id = [0]

    #the total is read from the last line
    prog = tqdm(desc='Seconds parsed', total=5011199)

    #consider either src or dst computer in redteam events
    fmt_label = lambda ts,computer : \
        1 if is_anomalous_node_range(anom_dict, computer, ts) \
        else 0

    fmt_u = lambda x : \
        x.split('@')[0].replace('$', '')

    #0:time,1:user@domain,2:computer,3:process name,4:start/end
    fmt_line = lambda ts,user,computer,process,status: (
        '%s,%s,%s,%s,%s,%s\n' % (
            ts, nmap[computer], umap[fmt_u(user)],get_or_add(process, proc_map, proc_id),
            get_or_add(status, status_map, status_id),fmt_label(int(ts),computer)
        ),
        int(ts)
    )

    while line:
        # Filtering out the rows with ?, likely data collection errors
        if '?' in line:
            line = f_in.readline()
            continue

        tokens = line.split(',')

        #Filtering out the lines with user or computer not in the mappings
        if not fmt_u(tokens[1]) in umap or not tokens[2] in nmap:
            line = f_in.readline()
            continue

        # last field has '\n', need to be removed
        l, ts = fmt_line(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4][:-1])

        if ts != last_time:
            prog.update(ts-last_time)
            last_time = ts

        # After ts progresses at least 10,000 seconds, make a new file
        # ZL: the last snapshot isn't saved
        if ts >= cur_time+DELTA:
            f_out.close()
            cur_time += DELTA
            f_out = open(proc_folder + str(cur_time) + '.txt', 'w')

        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(proc_map, 'procmap.pkl')
    save_map(status_map, 'statsmap.pkl')

if __name__ == '__main__':
    split_auth()
    #split_dns()
    #split_flows()
    # split_proc()
