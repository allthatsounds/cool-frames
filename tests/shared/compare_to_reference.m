function [n_pass, n_fail] = compare_to_reference(current, reference_path, tol)
% COMPARE_TO_REFERENCE  Compare current values against golden reference .mat.
%
%   [n_pass, n_fail] = compare_to_reference(current, reference_path)
%   [n_pass, n_fail] = compare_to_reference(current, reference_path, tol)
%
%   INPUTS
%     current        struct with current computed values (flat or nested)
%     reference_path path to golden .mat WITHOUT extension
%     tol            relative tolerance (default 1e-10)
%
%   OUTPUTS
%     n_pass  number of fields that matched
%     n_fail  number of fields that mismatched or were missing
%
%   Comparison uses relative error:
%     err = max(|cur - ref|) / (max(|ref|) + eps)
%
%   See also: save_reference

    if nargin < 3
        tol = 1e-10;
    end

    n_pass = 0;
    n_fail = 0;

    ref_file = [reference_path '.mat'];
    if ~exist(ref_file, 'file')
        warning('compare_to_reference:noFile', ...
            'Golden reference not found:\n  %s\nRun tests once to create it.', ref_file);
        return;
    end

    ref    = load(ref_file);
    fields = fieldnames(ref);

    fprintf('\n  Comparing against: %s\n', ref_file);
    fprintf('  %-40s  %s\n', 'Field', 'Result');
    fprintf('  %s\n', repmat('-', 1, 60));

    for k = 1 : numel(fields)
        fname = fields{k};

        if ~isfield(current, fname)
            fprintf('  %-40s  MISSING\n', fname);
            n_fail = n_fail + 1;
            continue;
        end

        cur_val = double(current.(fname));
        ref_val = double(ref.(fname));

        if ~isequal(size(cur_val), size(ref_val))
            fprintf('  %-40s  SIZE MISMATCH  cur=%s  ref=%s\n', ...
                fname, mat2str(size(cur_val)), mat2str(size(ref_val)));
            n_fail = n_fail + 1;
            continue;
        end

        max_ref = max(abs(ref_val(:)));
        err     = max(abs(cur_val(:) - ref_val(:))) / (max_ref + eps);

        if err < tol
            fprintf('  %-40s  OK  (rel_err=%.2e)\n', fname, err);
            n_pass = n_pass + 1;
        else
            fprintf('  %-40s  MISMATCH  rel_err=%.2e  (tol=%.2e)\n', fname, err, tol);
            n_fail = n_fail + 1;
        end
    end

    fprintf('  %s\n', repmat('-', 1, 60));
    fprintf('  Reference comparison: %d passed, %d failed\n\n', n_pass, n_fail);
end
