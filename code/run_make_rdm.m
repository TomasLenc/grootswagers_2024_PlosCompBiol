clear 

%%

nproc = 8; 
subjectnr = 1; 
%%


fn = sprintf('../../ds004357/derivatives/cosmomvpa/sub-%02i_task-rsvp_cosmomvpa.mat',subjectnr);

fprintf('loading %s\n',fn);tic
load(fn,'ds')
fprintf('loading data finished in %i seconds\n',ceil(toc))

% split the datasets based on the stimulus sequence rate (there were
% two)
ds_splits = cosmo_split(ds,'soaduration');


%% 

% let's just do the faster stimulus rate
split = 1; 


% get the data for the current stimulsu sequence rate
ds = ds_splits{split};  

fprintf('Making %.2fHz RDM\n',1/ds.sa.soaduration(1))

% added by tomo: speed up calculations and only get 3 timepoints
ds = cosmo_slice(ds, cosmo_match(ds.fa.time, [50, 100, 150]), 2); 

% split the 40 sequences into odd and even, and assign this as chunks
ds.sa.chunks = mod(ds.sa.sequencenumber,2);

% decoding target is the stimulus number code
ds.sa.targets = ds.sa.stimnumber;

% get unique targets (stimuli)
unique_targets = unique(ds.sa.targets);

% get unique chunks (which are just two here - odd and even sequences
% for this sequence rate), which will be used to train and test the
% classifiers respectively.
unique_chunks = unique(ds.sa.chunks);

% get unique pairwise combinations of stimuli (targets)
pairwise_target_combs = combnk(1:length(unique_targets), 2);

% go over chunks (a group of odd and even sequences)
mask_even_odd_seq = [];
for i_chunk = 1:length(unique_chunks)
    % look in the dataset and find which samples correspond to the
    % current chunk value - save that in a mask matrix where each row
    % corresponds to one sample, and each column encodes whether the
    % sample does/doesn't correcpond to the particular chunk
    mask_even_odd_seq(:,i_chunk) = ds.sa.chunks == unique_chunks(i_chunk);
end

% go over individual stimuli, and prepare a mask matrix. Each row is
% one sample in the dataset, and each column is a logical mask saying
% which of the unique targets (stimuli) the sample corresponds to
mask_target = [];
for e = 1:length(unique_targets)
    mask_target(:,e) = ds.sa.targets == unique_targets(e);
end

% make partitions (that will be used to run a crossvalidated classifier
% for each unique pair of stimuli)
partitions = struct('train_indices',[],'test_indices',[]);

sa = struct('target1',[],'target2',[],'lefoutchunk',[]);

cc = clock(); 
mm = '';

% go over all unique combinations of stimuli (target1 and target2 that
% will be decoded to build the RDM)
for i_target_comb = 1:length(pairwise_target_combs)

    % Now we have two stimuli (i.e. two targets) that we want to
    % decode. We need to find all samples in the dataset where either
    % one or the other target were presented. Let's use our big mask to
    % do that easily.
    mask_samples_with_either_target = any(mask_target(:, pairwise_target_combs(i_target_comb,:)), 2);

    % Now we will assign odd seuqnces as train and even sequences as
    % test in one iteration, and the reverse in the other iteration.
    for i_chunk = 1:length(unique_chunks)

        % save the indices of the samples that (i) came from odd
        % sequences and (ii) contain either of the stimulus pair we are
        % trying to decode as a cell
        partitions.train_indices{1,end+1} = find(mask_samples_with_either_target & ...
                                                 ~mask_even_odd_seq(:,i_chunk));

        % do the same with indices of samples that came from the even
        % sequences
        partitions.test_indices{1,end+1} = find(mask_samples_with_either_target & ...
                                                mask_even_odd_seq(:,i_chunk));


        % So now we have the current partition (which is essentially
        % defining a subset of samples that contain either of the
        % stimuli from the to-be-decoded pair, and half of them is
        % assigned for training, the other half for testing, based on
        % whether they came form the odd vs. even sequences). We just
        % want to save the information about what targets are going to
        % be decoded here.

        % Not that we're using unique combinations, so we don't have
        % the case that stimulus that's coded here as target1 and
        % target2 will be simply swapped in another pair - we only get
        % UNIQUE pairs from the combnk() function.

        % convince yourself: >> size(unique([pairwise_target_combs;
        % flip(pairwise_target_combs, 2)], 'rows')) there is no
        % repetition, which would happen if we had, e.g., [1, 2
        %        2, 1]
        % there the size would be the same after running the code above
        sa.target1(end+1,1) = unique_targets(pairwise_target_combs(i_target_comb,1));

        sa.target2(end+1,1) = unique_targets(pairwise_target_combs(i_target_comb,2));

        % this "leftoutchunk" is essentialy the chunk used for testing,
        % and corresponds either to odd or even half of the sequences.
        sa.lefoutchunk(end+1,1) = i_chunk;

    end

    % update progress bar
    if ~mod(i_target_comb,100)
        mm = cosmo_show_progress(cc,i_target_comb/length(pairwise_target_combs),...
                                 sprintf('%i/%i',i_target_comb,length(pairwise_target_combs)),...
                                 mm);
    end

end

% set up decoding

% put everthing we degined above into measure arguments for a
% crossvalidation function from cosmo
measure = @cosmo_crossvalidation_measure;

measure_args = {};
measure_args.partitions = partitions; 
measure_args.nproc = nproc; % number of processors
measure_args.classifier = @cosmo_classify_lda;
measure_args.check_partitions = 0;
measure_args.output = 'fold_accuracy';

% we have the MEG activity pattern separately for each timepoint. We
% want to get the timecourse, i.e. decodability of each stiulus pair
% separately for each single timepoint!!!! we can direcrtly make use of
% the cosmo searchlight function (which is obviously for much more than
% literal searchlight). To get one "seqrchlight" iteration for one
% single timepoint - just set hte neighbourhood "radius" to zero.
neighbourhood = cosmo_interval_neighborhood(ds,'time','radius',0);

% run "searchlight". this will run decoding using activity pattern
% across MEG sensors, for each "partition" we defined above, separately
% fort each timepoint
rdm = cosmo_searchlight(ds, neighbourhood, measure, measure_args);

% we have information about which stimulus was target1 and target2, and
% whether the even or odd sequences were used for testing, separately
% for each "partition". Let's just assign it. 
rdm.sa = cosmo_structjoin(rdm.sa, sa);


% save info about sequnce rate (for 20 Hz we did - this would be
% split=1 and soaduration=0.05)
rdm.sa.split = split * ones(size(rdm.samples,1), 1);
rdm.sa.soaduration = ds.sa.soaduration(1) * ones(size(rdm.samples,1),1);

% save info about the stimuli to disk (just average the samples in the
% dataset per stimulus, get the sample attributes - there will be one for
% each stimulus) 
x = cosmo_average_samples(cosmo_slice(ds, ds.sa.blocksequencenumber==1), ...
                          'split_by',{'stimnumber'});       
stimtable = struct2table(x.sa);
stimtable.f_colour = stimtable.f_color; % rename column for later

% save
outfn = sprintf('../results/sub-%02i_decoding_rdm_%ims.mat',subjectnr,1000*ds.sa.soaduration(1));
fprintf('saving %s\n',outfn);tic
save(outfn,'rdm','stimtable','-v7.3')
fprintf('saving data finished in %i seconds\n',ceil(toc))

%% build the actual RDM 

% we have 2 folds for each pair of stimuli (even seq train and off test,
% and the other way around) - let's average them 
rdm_avg = cosmo_average_samples(rdm,'split_by', {'target1','target2'});
% now we should have exactly half of samples in the rdm_avg

% vector of times
t = rdm_avg.a.fdim.values{1};

% build them 256x256 RDM fuckers separately for each timepoint 
X = nan(175, 256, 256);

% go sample by sample (which here corresponds to accuracy of decoding one
% unique pair of stimuli)
for i_sample = 1:length(rdm_avg.sa.target1)
    
    % target 1 index (need to add one cuase they start at 0)
    idx_target1 = 1 + rdm_avg.sa.target1(i_sample);
    idx_target2 = 1 + rdm_avg.sa.target2(i_sample);
    
    % use those indices to place the accuracy in the correct position in
    % the RDM matrix
    X(:, idx_target1, idx_target2) = rdm_avg.samples(i_sample, :);
    X(:, idx_target2, idx_target1) = rdm_avg.samples(i_sample, :);
    
end


figure
imagesc(squeeze(X(100, :, :)))

