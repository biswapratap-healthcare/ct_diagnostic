import pydicom
import concurrent.futures

from ct_slice import CTSlice


# def worker(inp):
#     ct_instance = inp
#     meta_data_dicom = pydicom.dcmread(ct_instance)
#     ct_slice = CTSlice(meta_data_dicom)
#     ggo, con, sub, fib, ple, pne, nor, affected_points, unaffected_points = ct_slice.get_box_label_distribution()
#     return ct_slice, ggo, con, sub, fib, ple, pne, nor, affected_points, unaffected_points


# def process(ct_instances):
#     rets = list()
#     print("Total number of instances to process : " + str(len(ct_instances)))
#     for ct_instance in ct_instances:
#         rets.append(worker(ct_instance))
#     return rets

def worker(inp):
    ct_instance = inp
    meta_data_dicom = pydicom.dcmread(ct_instance)
    ct_slice = CTSlice(meta_data_dicom)
    ggo, con, sub, fib, ple, pne, nor, affected_points = ct_slice.get_box_label_distribution()
    return ct_slice, ggo, con, sub, fib, ple, pne, nor, affected_points, meta_data_dicom


def process(ct_instances):
    inps = list()
    for ct_instance in ct_instances:
        inps.append(ct_instance)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        rets = executor.map(worker, inps)
        rets_list = list()
        for ret in rets:
            rets_list.append(ret)
        return rets_list


def process_2(ct_instances):
    for ct_instance in ct_instances:
        worker(ct_instance)
