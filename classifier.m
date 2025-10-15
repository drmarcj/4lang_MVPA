%% classifier script

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;



% define data filenames & load data from even and odd runs

data_path=fullfile('coefs'); % 

% l_fus_mask=fullfile('ROIs', 'FuG_L_resample.nii.gz'); % all of left fusiform anat
% vwfa_mask = fullfile('ROIs', 'vwfa_7.5mm.nii.gz'); % vwfa with a 7.5mm sphere
% occ_l_mask = fullfile('ROIs', 'OcG_L_resample.nii.gz'); % all of left (inferior) occipital
% occ_r_mask = fullfile('ROIs', 'OcG_R_resample.nii.gz'); % all of right (inferior) occipital
% r_fus_mask=fullfile('ROIs', 'FuG_R_resample.nii.gz'); % all of right fusiform
% r_ifg_mask=fullfile('ROIs', 'IFG_R_resample.nii.gz'); % R-IFG (just the three inferior bits I think)
% 
% l_fus_func_mask=fullfile('ROIs', 'FuG_func_mask.nii.gz'); % all of left fusiform U group reading & Speech network
% 
% l_fus_func_mask=fullfile('ROIs', 'FuG_L_func_mask.nii.gz'); % all of left fusiform U group reading network
% r_fus_func_mask=fullfile('ROIs', 'FuG_R_func_mask.nii.gz'); % all of left fusiform U group reading network
% l_ipl_func_mask=fullfile('ROIs', 'IPL_L_func.nii.gz');


% maskNames = {'AAL3_Angular_L_resample.nii.gz', 'AAL3_Fusiform_L_resample.nii.gz', 'AAL3_IFGL_resample.nii.gz',...
%     'AAL3_Inf_Parietal_L_resample.nii.gz', 'AAL3_Inf_Temporal_L_resample.nii.gz', 'AAL3_MFGL_resample.nii.gz',...
%     'AAL3_Mid_Temporal_L_resample.nii.gz', 'AAL3_SMAL_resample.nii.gz','AAL3_Sup_Parietal_L_resample.nii.gz',...
%     'AAL3_Sup_Temporal_L_resample.nii.gz', 'AAL3_Occipital_L_resample.nii.gz'}; 




maskNames = ...
{'AAL3_Inf_Temporal_L_resample_reading_conj.nii.gz',...
'AAL3_MFGL_resample_reading_conj.nii.gz',...
'AAL3_Mid_Temporal_L_resample_reading_conj.nii.gz',...
'AAL3_Occipital_L_resample_reading_conj.nii.gz',...
'AAL3_SMAL_resample_reading_conj.nii.gz',...
'AAL3_Sup_Parietal_L_resample_reading_conj.nii.gz',...
'AAL3_Sup_Temporal_L_resample_reading_conj.nii.gz',...
'AAL3_Angular_L_resample_reading_conj.nii.gz',...
'AAL3_Fusiform_L_resample_reading_conj.nii.gz',...
'AAL3_IFGL_resample_reading_conj.nii.gz',...
'AAL3_Inf_Parietal_L_resample_reading_conj.nii.gz'};

maskNames = {'FuG_L_func_mask.nii.gz','AAL3_Fusiform_L_resample.nii.gz'};

maskNames = { 'AAL3_visual_L_resample_reading.nii.gz'};




% anatomical-only masks
% maskNames = ...
% {'AAL3_Inf_Temporal_L_resample.nii.gz',...
% 'AAL3_MFGL_resample.nii.gz',...
% 'AAL3_Mid_Temporal_L_resample.nii.gz',...
% 'AAL3_Occipital_L_resample.nii.gz',...
% 'AAL3_Visual_L_resample.nii.gz'...    % nb this one is extremely generous in what we're calling 'visual'
% 'AAL3_SMAL_resample.nii.gz',...
% 'AAL3_Sup_Parietal_L_resample.nii.gz',...
% 'AAL3_Sup_Temporal_L_resample.nii.gz',...
% 'AAL3_Angular_L_resample.nii.gz',...
% 'AAL3_Fusiform_L_resample.nii.gz',...
% 'AAL3_IFGL_resample.nii.gz',...
% 'AAL3_Inf_Parietal_L_resample.nii.gz'};

% maskNames = {'AAL3_Fusiform_L_resample.nii.gz'};
% Kastner visual atlas ROIs, anatomical only
% maskNames = ...
%     {'kastnerV1v.nii.gz', 'kastnerV2v.nii.gz',...
%     'kastnerv3v.nii.gz','kastnerhv4.nii.gz',...    	
%     'kastnervo1.nii.gz','kastnervo2.nii.gz',...
%     'kastnerlo1.nii.gz','kastnerlo2.nii.gz'...
%     'kastnerto1.nii.gz','kastnerto2.nii.gz'}


% % Kastner visual atlas ROIs, functional 
% RH
  % maskNames = ...
  %     {'kastnerV1v_rh_reading_conj.nii.gz', 'kastnerV2v_rh_reading_conj.nii.gz',...
  %     'kastnerV3v_rh_reading_conj.nii.gz','kastnerv4v_rh_reading_conj.nii.gz'}
 
% LH
    maskNames = ...
      {'kastnerV1v_reading_conj.nii.gz', 'kastnerV2v_reading_conj.nii.gz',...
      'kastnerV3v_reading_conj.nii.gz','kastnerv4v_reading_conj.nii.gz',...
      'kastnerVO1VO2_reading_conj.nii.gz', 'kastnerLO1LO2_reading_conj.nii.gz'}


% combined Kastner ROIs:
%maskNames = ...
%    {'kastnerV1V2v.nii.gz','kastnerV3hV4v.nii.gz'...
%    'kastnerLO1LO2.nii.gz','kastnerVO1VO2.nii.gz'};

% bilateral combined Kastner ROIs:
%  maskNames = ...
%      {'kastnerV1V2v_bilateral.nii.gz','kastnerV3hV4v_bilateral.nii.gz'...
%      'kastnerLO1LO2_bilateral.nii.gz','kastnerVO1VO2_bilateral.nii.gz'};




% Kastner ROIs (coarse grained) combined with reading conjunction functional map
% maskNames = ...
%     {'kastnerV1V2v_reading_conj.nii.gz','kastnerV3hV4v_reading_conj.nii.gz'...
%     'kastnerLO1LO2_reading_conj.nii.gz','kastnerVO1VO2_reading_conj.nii.gz'};

% Kastner ROIs (coarse grained) bilateral, combined with reading conjunction functional map
%   maskNames = ...                                  
%       {'kastnerV1V2v_bilateral_reading_conj.nii.gz','kastnerV3hV4v_bilateral_reading_conj.nii.gz'...
%       'kastnerLO1LO2_bilateral_reading_conj.nii.gz','kastnerVO1VO2_bilateral_reading_conj.nii.gz'};
% 




% check filenames are valid to avoid throwing weird errors

for f=1:numel(maskNames)
    
    maskFile = fullfile('ROIs',maskNames{f});
    if isfile(maskFile)
         % File exists.
    else
         fprintf("Error: No such file: %s\n",maskFile);
         return;
    end
end


for iMask=1:numel(maskNames)

thisMask = maskNames{iMask};
fprintf("Processing mask: %s\n", thisMask);

maskFile = fullfile('ROIs',thisMask);
    
i=1;
%files = dir('coefs/*auditory.nii.gz');
files = dir('coefs/*visual.nii.gz');
for file = files'
    fn=fullfile(file.folder,file.name);
    ds{i}=cosmo_fmri_dataset(fn,'mask',maskFile,...
                            'targets',floor((i-1)/21)+1,'chunks',mod(i-1,21)+1);
    i=i+1;
end

ds_full = cosmo_stack(ds);
labels={'Spanish'; 'Hebrew'; 'Chinese'; 'English'};
ds_full.sa.labels = repelem(labels,21);
%ds_full = cosmo_normalize(ds_full,'demean',1);
cosmo_check_dataset(ds_full);

fprintf("%d features in mask\n",size(ds_full.samples,2))


%
%% Cross-validation classifier  
% Try each classifier, just for shits and giggles

partitions=cosmo_nfold_partitioner(ds_full);
test_pred = cosmo_crossvalidate(ds_full, @cosmo_classify_nn, partitions);
confusion_matrix_folds=cosmo_confusion_matrix(ds_full.sa.targets,test_pred);
confusion_matrix=sum(confusion_matrix_folds,3);
accuracy=sum(diag(confusion_matrix))/sum(confusion_matrix(:));
fprintf('NN: %.3f\n', accuracy)


partitions=cosmo_nfold_partitioner(ds_full);
test_pred = cosmo_crossvalidate(ds_full, @cosmo_classify_lda, partitions);
confusion_matrix_folds=cosmo_confusion_matrix(ds_full.sa.targets,test_pred);
confusion_matrix=sum(confusion_matrix_folds,3);
accuracy=sum(diag(confusion_matrix))/sum(confusion_matrix(:));
fprintf('LDA: %.3f\n', accuracy)

partitions=cosmo_nfold_partitioner(ds_full);
test_pred = cosmo_crossvalidate(ds_full, @cosmo_classify_naive_bayes, partitions);
confusion_matrix_folds=cosmo_confusion_matrix(ds_full.sa.targets,test_pred);
confusion_matrix=sum(confusion_matrix_folds,3);
accuracy=sum(diag(confusion_matrix))/sum(confusion_matrix(:));
fprintf('Bayes: %.3f\n', accuracy)

partitions=cosmo_nfold_partitioner(ds_full);
test_pred = cosmo_crossvalidate(ds_full, @cosmo_classify_svm, partitions);
confusion_matrix_folds=cosmo_confusion_matrix(ds_full.sa.targets,test_pred);
confusion_matrix=sum(confusion_matrix_folds,3);
accuracy=sum(diag(confusion_matrix))/sum(confusion_matrix(:));
fprintf('SVM: %.3f\n', accuracy);


% % feature-selection classifier using top 50% features
% partitions=cosmo_nfold_partitioner(ds_full);
% opt=struct();
% opt.child_classifier=@cosmo_classify_naive_bayes;
% opt.feature_selector=@cosmo_anova_feature_selector;
% opt.feature_selection_ratio_to_keep=.05;
% test_pred=cosmo_searchlight(ds_tl,nbrhood,measure,measure_args,...
%                                               'progress',false);

% @cosmo_classify_meta_feature_selection  

[test_pred,acc] = crossvalidate(ds_full, ...
                                     @cosmo_meta_feature_selection_classifier, ...
                                     partitions, opt);

confusion_matrix_folds=cosmo_confusion_matrix(ds_full.sa.targets,test_pred);
confusion_matrix=sum(confusion_matrix_folds,3);
fprintf('Bayes: %.3f\n', acc)




%% Cross-validation classification 
% I think this just does the same as the Bayes above, but differently.
%
%  Create a partitioner that will leave out the N-th subject in each
% language, so leave-4-out cross validation in our case

p = cosmo_nchoosek_partitioner(ds_full,1);


% create n-fold cross validation 
% http://cosmomvpa.org/matlab/run_nfold_crossvalidate.html#run-nfold-crossvalidate


nsamples=size(ds_full.samples,1); 
all_pred=zeros(nsamples,1);
test_targets= ds_full.sa.targets;
nfolds=numel(unique(ds_full.sa.chunks)); 

for fold=1:nfolds
    % make a logical mask (of size 60x1) for the test set. It should have
    % the value true where ds.sa.chunks has the same value as 'fold', and
    % the value false everywhere else. Assign this to the variable
    % 'test_msk'
    test_msk=ds_full.sa.chunks==fold;

    % slice the input dataset 'ds' across samples using 'test_msk' so that
    % it has only samples in the 'fold'-th chunk. Assign the result to the
    % variable 'ds_test';
    ds_test=cosmo_slice(ds_full,test_msk);

    % now make another logical mask (of size 60x1) for the training set.
    % the value true where ds.sa.chunks has a different value as 'fold',
    % and the value false everywhere else. Assign this to the variable
    % 'train_msk'
    train_msk=ds_full.sa.chunks~=fold;
    % (alternative: train_msk=~test_msk)

    % slice the input dataset again using train_msk, and assign to the
    % variable 'ds_train'
    ds_train=cosmo_slice(ds_full,train_msk);

    % Use cosmo_classify_lda to get predicted targets for the
    % samples in 'ds_test'. To do so, use the samples and targets
    % from 'ds_train' for training (as first and second argument for
    % cosmo_classify_lda), and the samples from 'ds_test' for testing
    % (third argument for cosmo_classify_lda).
    % Assign the result to the variable 'fold_pred', which should be a 6x1
    % vector.
    fold_pred=cosmo_classify_naive_bayes(ds_train.samples,ds_train.sa.targets,...
                                    ds_test.samples);

    % store the predictions from 'fold_pred' in the 'all_pred' vector,
    % at the positions masked by 'test_msk'.
    all_pred(test_msk)=fold_pred;
end
% 
% 
% %% Permutation test to estimate false positive rate
% tic;
% niter=10000;
% acc0=zeros(niter,1); % allocate space for permuted accuracies
% ds0=ds_full; % make a copy of the dataset
% classifier = @cosmo_classify_naive_bayes;
% partitions=cosmo_nfold_partitioner(ds_full);
% 
% % compute classification accuracy of the original data
% [pred, acc]=cosmo_crossvalidate(ds_full, classifier, partitions);
% 
% % for _niter_ iterations, reshuffle the labels and compute accuracy
% % Use the helper function cosmo_randomize_targets
% 
% for k=1:niter
%     ds0.sa.targets=cosmo_randomize_targets(ds_full);
%     [foo, acc0(k)]=cosmo_crossvalidate(ds0, classifier, partitions);
% end
% 
% toc;
% p=sum(acc<acc0)/niter;
% corrAlpha = prctile(acc0,95);
% fprintf('%d permutations: accuracy=%.3f, p=%.4f\n', niter, acc, p);
% bins=0:25/niter:1;
% h=histc(acc0,bins);
% bar(bins,h);
% hold on;
% %line([acc acc],[0,max(h)]);
% line([corrAlpha corrAlpha],[0,max(h)]);
% hold off;
% title(sprintf('a=.05: %.3f',corrAlpha));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate and plot confusion matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nclasses = numel(unique(test_targets));

confusion_matrix=cosmo_confusion_matrix(test_targets,all_pred);
% reorder to something sensible (Sp, En, He, Ch) and scale to prop correct
confusion_matrix = confusion_matrix([1 4 2 3],[1 4 2 3])/(nfolds);

disp(confusion_matrix);

confusion_matrix = (confusion_matrix'+ confusion_matrix)/2;

% Plot confusion matrix
figure
imagesc(confusion_matrix,[0 1]);
title(thisMask,'interpreter','none');
set(gca,'XTick',1:nclasses,'XTickLabel',labels([1 4 2 3]));
set(gca,'YTick',1:nclasses,'YTickLabel',labels([1 4 2 3]));
ylabel('target');
xlabel('predicted');
colormap('jet');
colorbar;

%save it
thisMask = thisMask(1:end-7);

fn = strcat('results/', thisMask,'-unnormed-visual.png');
saveas(gcf, fn);
end



