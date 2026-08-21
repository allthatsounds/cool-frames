function save_reference(data, filepath)
% SAVE_REFERENCE  Save a reference .mat file for later comparison.
%
%   save_reference(data, filepath)
%
%   Writes two files:
%     <filepath>_current.mat   always overwritten  (tracks current run)
%     <filepath>.mat           written ONLY if it does not yet exist
%                              (golden reference; never auto-overwritten)
%
%   INPUTS
%     data      struct whose fields are the values to save
%     filepath  full path WITHOUT extension
%                 e.g. '.../layer0_ops/reference/layer0_reference'
%
%   Python loading:
%     import scipy.io
%     d = scipy.io.loadmat('layer0_reference_current.mat', squeeze_me=True)
%
%   See also: compare_to_reference, make_signal_battery

    current_path   = [filepath '_current.mat'];
    reference_path = [filepath '.mat'];

    % Ensure directory exists
    refdir = fileparts(filepath);
    if ~isempty(refdir) && ~exist(refdir, 'dir')
        mkdir(refdir);
    end

    % Always save current run
    save(current_path, '-struct', 'data', '-v7');
    fprintf('  [save_reference] current run : %s\n', current_path);

    % Save golden reference only on first creation
    if ~exist(reference_path, 'file')
        save(reference_path, '-struct', 'data', '-v7');
        fprintf('  [save_reference] golden ref  : %s  (created)\n', reference_path);
    else
        fprintf('  [save_reference] golden ref  : exists, not overwritten\n');
    end
end
